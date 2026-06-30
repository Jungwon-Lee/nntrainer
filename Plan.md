# SmallThinker Sparse ReGLU FFN — Optimization B3 (hybrid repacked gate) + B1 (balanced active work)

## Status (2026-06-30)

- **B3 (hybrid repacked gate): DONE, kept.** 2-phase `compute_sparse_ffn`. x86 3x median:
  21B dense 19.4 → **B3 21.0 (+8%)**, 4B par. Better numerical match to dense + sets up the
  ARM win (repacked gate already has NEON; only up/down need a NEON kernel).
- **B1a (static even-split of active list): DONE, kept (default).** Optimal here.
- **B1b (dynamic work-stealing): DONE, REJECTED on x86 (−6~8%), kept behind `NNTR_MOE_CHUNK`
  env (default 0 = B1a).** Per-expert active work too small (~155 neurons) → atomic-chunk
  contention > the ~2.5% balancing headroom. The 18% per-thread busy-skew is mostly
  P/E-core speed, not recoverable idle. Revisit B1b on ARM big.LITTLE.
- **Next:** ARM NEON kernel for `q4_0_row_dot_q8` + `q4_0_row_axpy` (up/down only, thanks to
  B3) → Android S25 A/B (the paper's +65% edge regime).

## Context

The base-model sparse ReGLU FFN (`SmallThinkerSparseMoELayer`,
`smallthinker_moe_layer_base_sparse.{h,cpp}`) is implemented and validated: x86 21B
sparse **+6–13%** vs dense (relu_zero ~80%), 4B ~par, output coherent. It ports
PowerInfer `fused_sparse_moe`: per token, quantize `x→q8_0` once, then a **single
`parallel_for` over a static neuron range** where each thread runs the full
gate→ReLU→up→down chain into a per-thread buffer, reduced once with the routing weight.

**The dominant cost is the gate dot.** In `compute_sparse_ffn` every one of
`intermediate`(=768 for both 4B/21B) neurons computes a `gate·x` dot **regardless of
activation**, while only the ~20% active neurons add an `up·x` dot + `down` axpy. So per
expert the work is ~768 gate dots + ~150 up + ~150 down. The gate dots are currently
**plain per-row q4_0×q8_0** (`q4_0_row_dot_q8`), which is slower per byte than the
repacked `q4_0x8` GEMV the dense path uses (8-row SIMD, contiguous, better cache). Two
changes target this:

- **B3 (hybrid gate):** compute the *full* gate with the optimal **repacked dense GEMV**
  in one call, then run the sparse up/down only for active neurons. Moves the dominant
  cost onto the fast kernel; costs one extra barrier.
- **B1 (balanced active work):** the up/down work is data-dependent (~20% of neurons) and
  the current **static** neuron split can pile all active neurons onto one thread. Balance
  it (even split of a compacted active list, or dynamic chunking).

They interact: B3 separates gate from up/down, which makes B1 natural (B1 then balances
only the post-mask active up/down work). Implement **B3 first**, then **B1** on top.

Baselines to beat (x86, NNTR_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1, same long input,
both orders): 21B dense ~18.0–18.5 TPS, current sparse ~19–21 TPS; 4B dense 66.8, sparse
65.6. Bins: `Quick.AI/res/smallthinker/SmallThinker-21BA3B-Instruct/{Q4_0_dense_x86_base,
Q4_0_sparse_x86_base_v2}` and the 4B equivalents.

---

## B3 — Hybrid: repacked full gate + sparse up/down

### Recipe change (revert gate to repacked)

`tools/quantize_stream/quantize_stream_models.cpp` (and the Quick.AI copy) `writeMoeSparse`:
change the expert `_gate` write back from `writeTransposedMatrixPlain` →
`writeTransposedMatrix` (**repacked q4_0x8/x4**); keep `_up`/`_down` plain. (This is the
original layout before the gate→plain edit.) Regenerate `Q4_0_sparse_x86_base_v2`
(`--isa X86 --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0`) — and a 4B equivalent.

### Layer change (`compute_sparse_ffn`)

Restructure into two phases per token:

1. **Gate phase (dense, 1 barrier):** build a Tensor view of the token's `x`
   (`input.getSharedDataTensor({1,1,1,hidden}, token_offset)`) and `gate_out`
   (`{1,1,1,intermediate}`), then `token_view.dot(gate_proj, gate_out)` — this dispatches
   to the repacked q4_0×q8_0 GEMV (same path the base dense layer uses, internally
   parallelized + q8-quantizes `x`). Apply ReLU to `gate_out` (or fold into step 2's
   `g>0` test). `gate_proj` is now repacked, so it is **no longer read by `q4_0_row_dot_q8`**.
2. **Active up/down phase (1 barrier):** compact the active set
   `S = { j : gate_out[j] > 0 }` (serial scan over `intermediate`, ~µs), then
   `parallel_for` over `S` doing `up_j = q4_0_row_dot_q8(up_row_j, xq, nb)`,
   `h = gate_out[j]·up_j`, `q4_0_row_axpy(h, down_row_j, y_local[t], nb)` into per-thread
   buffers; reduce with the routing weight (unchanged). `xq` (q8 activation) is still
   needed for the up dots — quantize once per token as today.

**Barrier count:** 2/expert (gate GEMV + up/down) vs 1 today; still well below the prior
regressed design's 3+. **Expected win:** the ~768 plain gate dots become one optimal
repacked GEMV, and the parallel section shrinks to active-only up/down.

### Notes / risks

- The gate `.dot` internally quantizes `x→q8`; the up dots reuse our own `xq`. Minor
  double-quant of `x` (cheap); optionally share later.
- Numerical: gate now matches the dense repacked path exactly → ReLU mask closer to dense
  (good for token-id agreement). up/down unchanged.
- Verify `gate_out` ReLU semantics match (`g>0`), and that `token_view.dot` accepts the
  repacked q4_0 `gate_proj` (it does — same as base dense `compute_expert_forward`).

## B1 — Balanced active-neuron work distribution

Applies to B3's step-2 up/down loop (or, if B3 is skipped, to the current all-neuron loop).

- **B1a (default, simplest):** with B3's compacted active list `S`, split `S` **evenly**
  across threads (`[t·|S|/nth, (t+1)·|S|/nth)`). Because every element of `S` is active,
  this is inherently balanced — no dynamic machinery. This is the cleanest and likely
  sufficient.
- **B1b (if still imbalanced / no B3):** PowerInfer-style **dynamic work-stealing** — a
  shared `std::atomic<unsigned> next{0}`, threads `beg = next.fetch_add(CHUNK)` until
  `beg>=N`, processing `[beg, beg+CHUNK)`. Tune `CHUNK` (e.g. 64–128; `intermediate`=768 is
  small, so 256 is too coarse for 4–8 threads). Use over the all-neuron loop when active
  neurons aren't pre-compacted.

**Measurement gate:** before/after, log per-thread active-neuron counts and per-thread
busy time (extend `NNTR_RELU_SPARSITY_LOG`). Only keep B1b if B1a still shows >~15%
thread-time skew.

---

## Sequencing

1. **B3** (recipe revert + 2-phase layer) → regen 21B+4B sparse bins → A/B vs dense &
   current sparse. Keep if 21B improves beyond +13% (or 4B turns positive).
2. **B1a** (even split of active list) — essentially free given B3; measure.
3. **B1b** (dynamic chunking) — only if B1a leaves measurable skew.

Each step is independently measurable; keep whichever improves, revert regressions.

## Verification

- **Bench hygiene:** performance governor (or note powersave), `pkill -9 -f nntr_causallm`
  + `vmstat` idle ≈99%, `OPENBLAS_NUM_THREADS=1 NNTR_NUM_THREADS=4`, same long input, median
  of 3, **A/B both orders** (shared box noise).
- **Speed:** 21B sparse(B3) and sparse(B3+B1) vs dense and current sparse — report gen TPS.
  Also 4B (does the faster gate flip 4B to a win?).
- **Correctness:** output coherent; `relu_zero` unchanged (~80% 21B); active set `S` should
  now match the dense repacked gate's ReLU mask more closely than the plain-gate version.
- **Profiling check (B3):** confirm the parallel section shrank (gate moved out) and
  barrier count = 2/expert; (B1) per-thread busy-time skew reduced.

## Critical files

- `Applications/CausalLM/models/smallthinker/smallthinker_moe_layer_base_sparse.cpp`
  (`compute_sparse_ffn` 2-phase restructure; B1 split logic)
- `tools/quantize_stream/quantize_stream_models.cpp` (+ Quick.AI copy) — `writeMoeSparse`
  gate → `writeTransposedMatrix` (repacked)
- regen bins via the manually-built quantizer (scratchpad/quantize_stream); dense baseline
  `Q4_0_dense_x86_base`
- reference: `smallthinker_moe_layer.cpp` (base dense `token_input.dot(gate_proj,gate_out)`
  pattern to mirror for the gate phase); PowerInfer `fused_sparse_moe.cpp`
  (`current_neuron.fetch_add` dynamic chunking for B1b)
