# Introduce `KVCacheOptimizer` backend for configurable KV cache optimization

## Summary

We should refactor the current KV cache path so that KV cache storage and
optimization policies are managed by a dedicated `KVCacheOptimizer`. This will
make it possible to support runtime-selectable KV cache formats such as raw
FP16/UINT16, INT8, Q4, compression, eviction, and future paged attention.

The KV cache optimization policy should be selected from `nntr_config.json` when
the model is created.

## Background

Currently, `KVCacheManager` owns the cache buffers, but `mha_core` directly reads
and writes the actual cache tensors through external input slots:

- `input[3]`: key cache
- `input[4]`: value cache

This means `mha_core` still knows the physical cache layout and directly creates
write/read views internally. Adding an optimizer only next to `KVCacheManager`
would not affect the real MHA execution path unless `mha_core` is also routed
through the optimizer/runtime abstraction.

## Proposal

Add a dedicated `KVCacheOptimizer` abstraction responsible for:

- Allocating physical KV cache storage
- Appending new K/V entries
- Managing quantized or compressed cache formats
- Materializing read ranges for attention computation
- Managing scale/zero-point metadata
- Supporting save/load for backend-specific cache formats

`KVCacheManager` should remain responsible for:

- Cache lifecycle
- Position tracking
- Backend selection
- Model graph binding
- Save/load orchestration

`mha_core` should eventually stop assuming the physical cache layout and use a
cache runtime/adapter API instead.

## Runtime Configuration

KV cache optimization should be configured through `nntr_config.json`, since it
is a runtime execution and memory policy.

Example:

```json
{
  "kv_cache": {
    "backend": "int8",
    "format": "int8_per_token",
    "scale_granularity": "per_token_per_head",
    "materialize_dtype": "fp16",
    "fallback": "error"
  }
}
```

Default behavior:

- If `kv_cache` is absent, keep the current raw cache behavior.
- `backend=raw` should preserve existing behavior.
- Unsupported backend/platform combinations should either fail clearly or fall
  back to raw depending on `fallback`.

## Initial Implementation Plan

1. Add `KVCacheSpec` and `KVCacheConfig`.
2. Parse `nntr_config.json["kv_cache"]` during model setup.
3. Add `KVCacheOptimizer` abstract interface.
4. Implement `RawKVCacheOptimizer` first to preserve current behavior.
5. Move `KVCacheManager` storage ownership behind the optimizer backend.
6. Add an MHA cache access adapter so `mha_core` no longer directly assumes
   cache layout.
7. Add versioned save/load metadata for KV cache files.
8. Add an opt-in INT8 materialize-to-scratch backend.
9. Later, add fused INT8/Q4 attention kernels for decode performance.

## Why INT8 First

INT8 is a safer first quantized backend than Q4 because:

- Streaming append is simpler.
- Scale metadata is easier to manage.
- Materialize-to-scratch implementation is straightforward.
- Accuracy regression is easier to debug.
- Existing MHA kernels can initially be reused.

## Acceptance Criteria

- Existing raw KV cache behavior remains unchanged by default.
- `nntr_config.json` can select the KV cache backend.
- Raw backend tests pass through the new optimizer abstraction.
- INT8 backend can be enabled opt-in and produces outputs within an agreed
  tolerance.
- Save/load includes enough metadata to distinguish raw and optimized cache
  formats.
- `mha_core` no longer directly depends on raw cache layout in the optimized
  path.

## Open Questions

- Should `mha_core` receive the optimizer runtime through explicit layer handle
  injection or a registry?
- Should `backend` and `format` be separate config fields?
- What should the default unsupported-backend policy be: `error` or raw
  fallback?
- What scale granularity should INT8 use first: per-token, per-token-per-head,
  or per-group?
- Should legacy raw precomputed cache files remain loadable?
