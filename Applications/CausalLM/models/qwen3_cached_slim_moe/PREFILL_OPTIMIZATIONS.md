# Qwen3 Cached Slim MoE Prefill Optimizations

This branch combines the Qwen3 Cached Slim MoE prefill work based on
`origin/main` commit `308bdd4f3b87a9dd5d5899104be32e6518efd637`.

## Motivation

A long prompt routes at least one token to most experts. The previous forward
path mapped every active expert before evicting the cache overflow, so peak
resident expert weights and synchronous storage I/O grew with the number of
experts reached by the prompt. It also retained per-expert outputs and created
temporary tensors repeatedly.

## Integrated changes

### Bounded expert residency

- Plan the final 32-expert LRU cache before expert computation.
- Evict experts outside that plan first.
- Keep only planned experts mapped after the forward call.
- Map other active experts transiently and release them immediately after use.
- Use an RAII lease so partial activation or compute failures do not leak
  mappings.

The steady cache is capped at 32 experts. During computation, at most the
current transient expert and one prefetched transient expert can be mapped in
addition to the steady cache.

### One-expert lookahead prefetch

`moe_prefetch_distance` controls virtual-weight lookahead:

- `1` (default): activate the next expert miss before computing the current
  expert.
- `0`: disable lookahead.

Only one future miss is activated. On Linux and Android, activation also issues
`MADV_WILLNEED`, allowing storage page-in to overlap the current expert's
compute without returning to the previous all-expert mapping behavior.

### Router work reduction

- Run `topK` directly on raw router logits.
- Apply softmax only to the selected logits, which preserves Qwen3
  `norm_topk_prob` results while avoiding exponentiation of unselected experts.
- Remove the second `topK(topk + 5)` pass and the unused expert-mask tensor.
- Derive cache recency from the actual routed top-k indices.
- Use an allocation-free small-k topK path for the common MoE case.

### Forward-memory reduction

- Allocate expert intermediate scratch tensors once per forward call and reuse
  views sized to each expert's assigned tokens.
- Keep only one compact expert output tensor live at a time.
- Scatter-add that output immediately, preserving the ascending expert
  accumulation order.

Peak temporary output storage changes from the sum of all active experts'
outputs to the largest single active expert output. Scratch storage changes
from repeated per-expert allocation to four reusable maximum-capacity buffers.

### Virtual tensor mapping portability

Virtual tensor `mmap`/`munmap` alignment now uses the runtime system page size
instead of assuming 4096 bytes. This is required for platforms with larger
pages and makes the prefetch activation path fail safely before publishing an
invalid mapping.

## Expected impact

The hard residency limit is expected to provide the largest improvement for
long prompts that touch more experts than the cache can hold, because it avoids
mapping all active experts simultaneously. Streaming outputs and scratch reuse
primarily reduce allocator and memory-pressure overhead. Router changes reduce
CPU work, while lookahead prefetch targets the remaining expert page-in stalls.

Actual speedup depends on prompt routing distribution, model precision,
storage, page-cache state, and platform support for `MADV_WILLNEED`; benchmark
cold-cache and warm-cache prefill separately.

## Validation coverage

- Runtime-page-size mapping alignment and activation failure handling.
- Allocation-free small-k topK results, ties, NaNs, and output reuse.
- Recent-expert collection and cache planning.
- Selected-logit softmax equivalence to normalized full softmax.
- Tiny Cached Slim MoE model construction and greedy generation.

The cache capacity remains expert-count based. A future byte-budget policy
should account for each expert's gate, up, and down tensor sizes so quantized
and FP32 models obey the same memory budget.
