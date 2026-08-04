# SmallThinker Model

This directory contains the SmallThinker Causal LM integration.

Supported architecture:

- `SmallThinkerForCausalLM`
- `SmallThinkerSlimForCausalLM`

The implementation follows SmallThinker's MoE decoder shape: GQA attention,
top-k primary routing, ReLU-gated experts, and a separate pre-attention router
input for the MoE block.

- `smallthinker_moe_layer.cpp`: SmallThinker-specific MoE layer with separate
  expert and router inputs.
- `smallthinker_moe_layer_slim.cpp`: Slim MoE variant that maps expert weights
  on demand and releases them after each active expert is computed.

Supported model configs:

- `SmallThinker-4BA0.6B-Instruct`: full RoPE attention layout.
- `SmallThinker-21BA3B-Instruct`: hybrid NoPE/RoPE and sliding-window layout
  from `rope_layout` and `sliding_window_layout`.

## Sparse LM head

SmallThinker's vocabulary predictor can skip inactive Q4_0 LM-head rows during
single-token decoding. Enable it in `nntr_config.json`:

```json
{
  "sparse_lmhead": true,
  "predictor_unit": 128,
  "lmhead_predictor_threshold": -2.0,
  "lmhead_predictor_topk_floor": 0,
  "lmhead_dtype": "Q4_0"
}
```

The model binary must contain three tensors at the end, in this order:

1. an explicit LM-head weight, including for tied-embedding models;
2. predictor `profiler_w1` with shape `[hidden_size, predictor_unit]`;
3. predictor `profiler_w2` with shape `[predictor_unit, vocab_size]`.

All three tensors use NNTrainer's standard architecture-specific Q4_0 repack:
Q4_0x4 on ARM and Q4_0x8 on x86. A binary generated for one architecture must
therefore not be reused on the other architecture.

Package the predictor from the official `model_lm_head.pt` together with a
standard FP32 SmallThinker model as follows:

```bash
python3 Applications/CausalLM/models/smallthinker/extract_lmhead_predictor.py \
  /path/to/model_lm_head.pt /tmp/smallthinker_predictor_fp32.bin

./build/Applications/CausalLM/nntr_quantize /path/to/smallthinker-fp32 \
  --output /path/to/smallthinker-q4-sparse \
  --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 \
  --sparse_lmhead \
  --predictor_fp32 /tmp/smallthinker_predictor_fp32.bin \
  --isa X86
```

Use `--isa ARM` for an ARM deployment. The quantizer validates the predictor
dimensions, appends an explicit head for tied-embedding models, repacks all
three sparse-head tensors for the selected ISA, and writes the predictor
settings into the generated `nntr_config.json`. The source must be the standard
`FP32-FP32` model rather than an already sparse or quantized model.

`NNTR_SPARSE_LMHEAD=0` disables candidate filtering at runtime.
`NNTR_LMHEAD_THRESHOLD` and `NNTR_LMHEAD_TOPK_FLOOR` override the JSON values.
Set `NNTR_LMHEAD_SPARSITY_LOG=1` to compare the selected vocabulary against the
dense argmax while tuning the predictor.
