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
