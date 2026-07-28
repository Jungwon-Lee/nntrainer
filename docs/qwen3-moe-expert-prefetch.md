# Qwen3 MoE Expert Prefetch Plan

## Status

- Target: Android arm64 first, Linux x86-64 second
- Scope: Qwen3 Slim MoE and Qwen3 Cached-Slim MoE inference
- Current phase: Phase 1 implementation complete; device performance
  validation pending
- Phase 0 implementation:
  - [x] Runtime page-size mapping calculation
  - [x] Page-unaligned virtual tensor lifecycle test
  - [x] 16 KB-page macOS arm64 compile validation
  - [ ] Android 4 KB device/system-image validation
  - [ ] Android 16 KB device/system-image validation
  - [ ] Linux x86-64 validation
- Phase 1 implementation:
  - [x] Android/Linux `MADV_WILLNEED` on successful virtual tensor mappings
  - [x] Slim active-expert batch activation and exception cleanup
  - [x] Cached-Slim active hit/miss grouping
  - [x] Cached-Slim miss activation before hit and miss compute
  - [x] Original expert-output accumulation order retained
  - [x] Prefetch, hit compute, miss compute, and eviction debug timers
  - [ ] Android cold/warm-cache performance validation
  - [ ] Linux x86-64 performance validation

## Background

Qwen3 Slim MoE keeps expert weights as virtual tensors backed by the model
file. An expert weight is activated with `mmap()` immediately before its
GEMV/GEMM and deactivated after use. Cached-Slim keeps recently used mappings
active and only maps cache misses.

`mmap()` does not copy the complete weight into memory. The first GEMV/GEMM
access can still trigger page faults and storage reads. The prefetch work aims
to issue those reads earlier and overlap them with useful computation.

The initial implementation will use the kernel page cache and
`MADV_WILLNEED`. It will not introduce an asynchronous I/O thread pool.

## Goals

1. Support Android devices using either 4 KB or 16 KB memory pages.
2. Issue prefetch hints for all active cache-miss experts before computing
   those experts.
3. Compute cache-hit experts while miss prefetch requests are in flight.
4. Preserve the existing output accumulation order and numerical behavior.
5. Keep Linux x86-64 behavior compatible for local development and
   benchmarking.
6. Add enough instrumentation to separate mapping, hit compute, miss compute,
   and eviction costs.

## Non-goals for the first implementation

- A dedicated asynchronous I/O thread pool
- `io_uring`, Linux `readahead()`, or Android-specific private APIs
- Explicit CPU page touching or `MADV_POPULATE_READ`
- Speculative prefetch of inactive experts
- Changing the converted model file format
- Windows virtual tensor support
- GPT-OSS prefetch changes

## Phase 0: 16 KB page-size support

### Problem

Virtual tensor activation currently aligns the model file offset to a
hard-coded 4096-byte boundary:

```cpp
size_t off = (file_offset / 4096) * 4096;
```

Android 15 supports devices configured with a 16 KB page size. `mmap()` file
offsets must be aligned to the runtime page size, so a 4 KB-aligned offset can
fail with `EINVAL` on such a device.

Android recommends using `getpagesize()` or `sysconf(_SC_PAGESIZE)` and
removing hard-coded 4096-byte assumptions:

<https://developer.android.com/guide/practices/page-sizes>

### Change

Both `Tensor::activate()` and `Tensor::deactivate()` will calculate their
mapping range with the runtime page size:

```text
page_size = sysconf(_SC_PAGESIZE)
map_offset = floor(file_offset / page_size) * page_size
data_offset = file_offset - map_offset
map_length = tensor_bytes + data_offset
```

Activation and deactivation must share the same calculation so that
`munmap()` receives the exact address and length used by `mmap()`.

`Tensor::activate()` also keeps a new mapping in a local variable until both
`mmap()` and the underlying tensor activation succeed. A failed `mmap()` is
therefore never stored as tensor state, and a later activation failure rolls
back the new mapping.

### Validation

- Host unit test with a virtual tensor at a page-unaligned file offset
- Activate, validate the mapped values, deactivate, and reactivate
- Run the same test on Android 4 KB and Android 16 KB system images
- Verify Linux x86-64 behavior

The test offset will be selected so that the previous 4096-byte calculation
is invalid on a 16 KB device. This makes the Android 16 KB run a regression
test for the original failure.

## Phase 1: active-miss advisory prefetch

### Current sequence

```text
route
  |
  +-- expert 0: mmap -> first-touch page faults -> compute
  +-- expert 1: mmap -> first-touch page faults -> compute
  +-- expert 2: mmap -> first-touch page faults -> compute
```

### Proposed sequence

```text
route
  |
  +-- identify cache hits and misses
  |
  +-- mmap + MADV_WILLNEED for every active miss
  |
  +-- compute cache hits
  |
  +-- compute cache misses
  |
  +-- accumulate outputs in the original expert order
```

`MADV_WILLNEED` is a best-effort hint and does not guarantee that pages are
resident when the call returns. Issuing all active-miss hints first gives the
kernel time to read later experts while an earlier expert or a cache hit is
being computed.

On Linux, `POSIX_FADV_WILLNEED` is also documented to initiate a nonblocking
read into the page cache. It is a possible follow-up if mapping-based advice
is insufficient:

<https://man7.org/linux/man-pages/man2/posix_fadvise.2.html>

### Why the compute ThreadManager is not used

The current `ThreadManager::parallel_for()` is synchronous and does not
provide an enqueue-and-return interface. Expert GEMV/GEMM kernels already use
the same manager. Running expert compute from a ThreadManager worker would
create a forbidden nested `parallel_for()` and can deadlock.

The first implementation therefore issues mapping and advice calls on the
inference thread and relies on the kernel for asynchronous storage work.

### Expert representation

Prefetch code must operate on stable references or pointers to the three
expert tensors:

```cpp
struct ExpertWeights {
  int expert_idx;
  Tensor *gate;
  Tensor *up;
  Tensor *down;
};
```

It must not copy an activated `Tensor` by value. The current Tensor copy
constructor copies `mapped_ptr`, while every virtual Tensor destructor can
call `deactivate()`. Copying an active Tensor can therefore create ambiguous
mapping ownership and duplicate `munmap()` attempts.

### Exception safety

Activating one expert is a three-step transaction. If the second or third
mapping fails, mappings created by that transaction must be rolled back.
Mappings that were already active before the transaction must remain active.

Cache metadata will only be committed after all required mappings for the
expert have succeeded.

### Cached-Slim ordering

Cached-Slim will:

1. Build the active expert list in the existing order.
2. Split the list into hits and misses.
3. Activate and advise all misses.
4. Commit successful mappings to the cache.
5. Compute hits first.
6. Compute misses.
7. Combine `expert_outputs` in the original active expert order.
8. Evict only after all expert computation is complete.

Changing compute order is safe because each expert writes to a separate
output tensor. The final accumulation order will not change.

### Slim ordering

Slim has no persistent expert cache. It will:

1. Collect all active experts.
2. Activate and advise all active experts.
3. Compute all active experts.
4. Deactivate all active experts, including exception cleanup.

This temporarily keeps more mappings active than the current
map-compute-unmap loop. Only the active top-k experts are included in the
first implementation.

## Phase 2: optional asynchronous I/O manager

This phase will only proceed if Phase 1 shows useful page-fault movement but
insufficient overlap.

A separate bounded I/O manager, initially with one worker, could perform
blocking prefault work while the normal ThreadManager computes cache hits.
The manager must provide:

- Priority for active misses over speculative work
- Duplicate request coalescing
- Byte-based in-flight limits
- Exception delivery to the inference thread
- Cancellation and shutdown
- Protection against eviction while an expert is prefetching or in use

This manager must remain separate from the compute ThreadManager.

## Instrumentation

Cached-Slim debug output now separates:

- Active hit and miss counts
- Prefetch issue time
- Cache-hit compute time
- Cache-miss compute time
- Eviction time
- Total MoE layer time

Byte and page-fault counters remain follow-up work. Prefetch issue time is the
wall-clock time spent mapping and submitting advice; it does not mean all
pages are resident when the timer stops.

Android benchmarks should additionally collect:

- Minor and major page faults
- Token latency, including p50 and p95
- First-token latency and steady-state throughput
- Process RSS/PSS
- Storage read throughput
- Thermal state where available

## Benchmark matrix

| Variant | Behavior |
| --- | --- |
| Baseline | Map and immediately compute each expert |
| Active prefetch | Map/advise all active misses, then compute |
| Hit overlap | Map/advise misses, compute hits, then compute misses |

Measurements must include both a cold-process run and a warm-cache run. A
prefetch change is acceptable when it improves cold miss latency without a
material warm-cache regression or excessive PSS/read amplification.

## Commit structure

1. `[tensor] Support runtime page size for virtual tensor mmap`
   - Replace the 4096-byte assumption
   - Share mapping-range calculation between activate/deactivate
   - Add virtual tensor mmap lifecycle tests
2. `[CausalLM] Prefetch active Qwen3 MoE experts before compute`
   - Add active expert grouping
   - Issue all miss advice before compute
   - Compute hits before misses
   - Preserve output order and add rollback
   - Add cumulative phase metrics

## Acceptance criteria

- Correct inference output is unchanged.
- Virtual tensor activation works on Android 4 KB and 16 KB page-size images.
- Linux x86-64 unit tests pass.
- No nested ThreadManager invocation is introduced.
- No mapping is leaked on partial activation failure.
- No expert is unmapped while compute is using it.
- Warm-cache performance does not materially regress.

## Current validation result

- `git diff --check`: passed
- clang-format 14 read-only format check: passed
- Tensor implementation compiled with the repository's `-Werror` flags
- Virtual tensor lifecycle test source compiled
- Qwen3 Slim and Cached-Slim implementation files compiled with transformer
  support and the repository's `-Werror` flags
- Full macOS unit-test linking is currently blocked by the existing
  unconditional `malloc.h` use in `nntrainer/tensor/swap_device.cpp`
- Android validation is pending because no Android NDK or connected device is
  available in the current workspace
