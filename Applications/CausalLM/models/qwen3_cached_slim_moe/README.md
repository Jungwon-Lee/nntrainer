# Qwen3 Cached Slim MoE Model

This directory contains the implementation for Qwen3 Slim MoE model with caching support.

> 📌 **Note** on `Cached-Slim`: This model extends the Slim approach (dynamic loading) by caching active experts. This strategy minimizes storage I/O bottlenecks, offering a sweet spot between low memory footprint and high inference speed.

## Files
- `qwen3_cached_slim_moe_causallm.cpp`: Cached Slim MoE implementation.
- `qwen_moe_layer_cached.cpp`: Cached MoE layer implementation.
- `PREFILL_OPTIMIZATIONS.md`: Integrated long-context prefill optimizations,
  cache bounds, configuration, and validation coverage.

## Cache sizing follow-up

The current cache capacity is expressed as an expert count. A byte-budget
property should be implemented together with hard cache admission and eviction,
not independently: the accounting must sum `getMemoryBytes()` for each
expert's gate, up, and down weights so FP32 and quantized weights obey the same
memory limit. The existing expert-count setting should remain as a compatibility
fallback.

Gate/up weight fusion is intentionally deferred because it requires a serialized
weight-layout change. It should be evaluated as a separate model-format
migration with backward compatibility.
