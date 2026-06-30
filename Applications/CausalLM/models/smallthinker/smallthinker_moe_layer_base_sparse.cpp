/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_moe_layer_base_sparse.cpp
 * @date   30 June 2026
 * @brief  SmallThinker BASE MoE layer with ReLU (ReGLU) activation sparsity.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @note   Ported from PowerInfer fused_sparse_moe (Tiiny-AI/PowerInfer,
 *         smallthinker/powerinfer/fused_sparse_moe/fused_sparse_moe.cpp). Each
 *         token's expert FFN computes the gate in full to obtain the ReLU mask,
 *         then the up/down projections only for active neurons. Threads split
 *         the neuron range and accumulate into per-thread output buffers (one
 *         barrier per token), reduced once at the end with the routing weight.
 *
 *         Expert gate/up/down are PLAIN per-neuron-row Q4_0 [intermediate,
 *         hidden]; row j of each starts at j * (hidden/32) * 18 bytes. The
 *         activation is quantized to Q8_0 once per token, and gate/up are
 *         computed with a fused Q4_0 x Q8_0 integer dot (no fp32 dequant temp,
 *         matching PowerInfer vec_dot_q4_0_q8_0); down is a fused dequant+axpy.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cpu_backend.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <smallthinker_moe_layer_base_sparse.h>
#include <smallthinker_sparse_ffn.h>
#include <thread_manager.h>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace causallm {

namespace {

// Plain Q4_0 weight block: fp16 delta (2B) + 16 nibble bytes (32 weights) = 18B
// (matches the .bin layout, produced by quantize_stream).
constexpr size_t kQ4_0_D = sizeof(uint16_t);
constexpr size_t kQ4_0_BLOCK = kQ4_0_D + 16;
// Activation Q8_0 block (produced AND consumed here, so the layout is ours):
// fp32 delta (4B) + 32 int8 quants = 36B. fp32 delta avoids an fp32->fp16 round
// and a dependency on nntrainer's (un-exported) quantize_row_q8_0<float>.
constexpr size_t kQ8_0_D = sizeof(float);
constexpr size_t kQ8_0_BLOCK = kQ8_0_D + 32;

// Quantize one fp32 row to our Q8_0 activation layout, block by block.
static inline void quantize_row_q8_0_local(const float *x, uint8_t *q,
                                           unsigned hidden) {
  const unsigned nb = (hidden + 31) / 32;
  for (unsigned b = 0; b < nb; ++b) {
    const float *xb = x + b * 32;
    float amax = 0.0f;
    for (unsigned i = 0; i < 32; ++i) {
      const float a = std::fabs(xb[i]);
      if (a > amax)
        amax = a;
    }
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    uint8_t *qb = q + b * kQ8_0_BLOCK;
    std::memcpy(qb, &d, sizeof(float));
    int8_t *qs = reinterpret_cast<int8_t *>(qb + kQ8_0_D);
    for (unsigned i = 0; i < 32; ++i)
      qs[i] = (int8_t)std::lround(xb[i] * id);
  }
}

// Exact IEEE-754 binary16 -> binary32 (lossless; matches nntr_half storage).
static inline float fp16_to_fp32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t man = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign; // +/- 0
    } else {
      // subnormal half -> normalized float
      exp = 1;
      while ((man & 0x400) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      out = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000u | (man << 13); // inf / nan
  } else {
    out = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

// Fused Q4_0 x Q8_0 dot of one weight row (nb blocks) against a q8_0-quantized
// activation. No fp32 dequant temp: integer MAC over packed nibbles, matching
// ggml/PowerInfer vec_dot_q4_0_q8_0.
static inline float q4_0_row_dot_q8(const uint8_t *w, const uint8_t *xq,
                                    unsigned nb) {
  float acc = 0.0f;
#if defined(__AVX2__)
  __m256 vacc = _mm256_setzero_ps();
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m256i off = _mm256_set1_epi8(8);
  for (unsigned b = 0; b < nb; ++b) {
    const uint8_t *q4 = w + b * kQ4_0_BLOCK;
    const uint8_t *q8 = xq + b * kQ8_0_BLOCK;
    const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t *>(q4)) *
                    *reinterpret_cast<const float *>(q8);
    // 16 packed nibble bytes -> 32 signed weights (-8..7): low nibbles = first
    // 16 weights, high nibbles = next 16 (matches dequantize_row_q4_0 layout).
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q4 + kQ4_0_D));
    const __m128i lo = _mm_and_si128(bytes, low_mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
    __m256i qx = _mm256_set_m128i(hi, lo);          // 32 x uint8 (0..15)
    qx = _mm256_sub_epi8(qx, off);                  // 32 x int8 (-8..7)
    const __m256i qy =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + kQ8_0_D));
    // signed x signed dot via abs/sign trick (maddubs wants unsigned * signed)
    const __m256i ax = _mm256_sign_epi8(qx, qx);    // |qx|
    const __m256i sy = _mm256_sign_epi8(qy, qx);    // qy * sign(qx)
    const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    vacc = _mm256_add_ps(
      vacc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32)));
  }
  // horizontal sum of vacc
  __m128 lo128 = _mm256_castps256_ps128(vacc);
  __m128 hi128 = _mm256_extractf128_ps(vacc, 1);
  lo128 = _mm_add_ps(lo128, hi128);
  lo128 = _mm_add_ps(lo128, _mm_movehl_ps(lo128, lo128));
  lo128 = _mm_add_ss(lo128, _mm_shuffle_ps(lo128, lo128, 0x1));
  acc = _mm_cvtss_f32(lo128);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  float32x4_t vacc = vdupq_n_f32(0.0f);
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);
  const int8x16_t off = vdupq_n_s8(8);
  for (unsigned b = 0; b < nb; ++b) {
    const uint8_t *q4 = w + b * kQ4_0_BLOCK;
    const uint8_t *q8 = xq + b * kQ8_0_BLOCK;
    const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t *>(q4)) *
                    *reinterpret_cast<const float *>(q8);
    const uint8x16_t bytes = vld1q_u8(q4 + kQ4_0_D);
    // low nibbles -> weights 0..15, high nibbles -> weights 16..31 (-8 offset)
    const int8x16_t lo =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, low_mask)), off);
    const int8x16_t hi =
      vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), off);
    const int8x16_t q8lo = vld1q_s8(reinterpret_cast<const int8_t *>(q8 + kQ8_0_D));
    const int8x16_t q8hi =
      vld1q_s8(reinterpret_cast<const int8_t *>(q8 + kQ8_0_D + 16));
    int32x4_t sumi = vdotq_s32(vdupq_n_s32(0), lo, q8lo);
    sumi = vdotq_s32(sumi, hi, q8hi);
    vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(sumi), d);
  }
  acc = vaddvq_f32(vacc);
#else
  for (unsigned b = 0; b < nb; ++b) {
    const uint8_t *q4 = w + b * kQ4_0_BLOCK;
    const uint8_t *q8 = xq + b * kQ8_0_BLOCK;
    const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t *>(q4)) *
                    *reinterpret_cast<const float *>(q8);
    const uint8_t *qs4 = q4 + kQ4_0_D;
    const int8_t *qs8 = reinterpret_cast<const int8_t *>(q8 + kQ8_0_D);
    int sumi = 0;
    for (unsigned i = 0; i < 16; ++i) {
      const int lo = (int)(qs4[i] & 0x0F) - 8;
      const int hi = (int)(qs4[i] >> 4) - 8;
      sumi += lo * (int)qs8[i] + hi * (int)qs8[i + 16];
    }
    acc += d * (float)sumi;
  }
#endif
  return acc;
}

// Fused dequant + scaled accumulate of one Q4_0 weight row into y (y += scale *
// dequant(row)). No fp32 temp row.
static inline void q4_0_row_axpy(float scale, const uint8_t *w, float *y,
                                 unsigned nb) {
#if defined(__ARM_NEON)
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);
  const int8x16_t off = vdupq_n_s8(8);
  // Widen an int8x16 of weights to 4 float32x4 and FMA into y[dst..dst+15].
  auto fma16 = [](float *dst, int8x16_t v, float d) {
    const int16x8_t l = vmovl_s8(vget_low_s8(v));
    const int16x8_t h = vmovl_s8(vget_high_s8(v));
    float32x4_t y0 = vld1q_f32(dst + 0), y1 = vld1q_f32(dst + 4);
    float32x4_t y2 = vld1q_f32(dst + 8), y3 = vld1q_f32(dst + 12);
    y0 = vmlaq_n_f32(y0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))), d);
    y1 = vmlaq_n_f32(y1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))), d);
    y2 = vmlaq_n_f32(y2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))), d);
    y3 = vmlaq_n_f32(y3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))), d);
    vst1q_f32(dst + 0, y0);
    vst1q_f32(dst + 4, y1);
    vst1q_f32(dst + 8, y2);
    vst1q_f32(dst + 12, y3);
  };
  for (unsigned b = 0; b < nb; ++b) {
    const uint8_t *q4 = w + b * kQ4_0_BLOCK;
    const float d = scale * fp16_to_fp32(*reinterpret_cast<const uint16_t *>(q4));
    const uint8x16_t bytes = vld1q_u8(q4 + kQ4_0_D);
    const int8x16_t lo =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, low_mask)), off);
    const int8x16_t hi =
      vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), off);
    float *yb = y + b * 32;
    fma16(yb, lo, d);      // weights 0..15
    fma16(yb + 16, hi, d); // weights 16..31
  }
#else
  for (unsigned b = 0; b < nb; ++b) {
    const uint8_t *q4 = w + b * kQ4_0_BLOCK;
    const float d = scale * fp16_to_fp32(*reinterpret_cast<const uint16_t *>(q4));
    const uint8_t *qs4 = q4 + kQ4_0_D;
    float *yb = y + b * 32;
    for (unsigned i = 0; i < 16; ++i) {
      yb[i] += d * (float)((int)(qs4[i] & 0x0F) - 8);
      yb[i + 16] += d * (float)((int)(qs4[i] >> 4) - 8);
    }
  }
#endif
}

constexpr unsigned kMaxThreads = 64;

// Optional ReLU-sparsity instrumentation (env NNTR_RELU_SPARSITY_LOG): counts
// neurons examined vs activated; printed once at process exit. With
// NNTR_MOE_THREAD_PROFILE it also reports per-thread busy-time skew of the
// up/down phase (to decide whether B1b dynamic balancing is worthwhile).
struct SparsityStats {
  std::atomic<size_t> examined{0};
  std::atomic<size_t> activated{0};
  std::atomic<long long> busy_ns[kMaxThreads] = {};
  ~SparsityStats() {
    size_t ex = examined.load();
    if (ex == 0)
      return;
    std::fprintf(stderr,
                 "[smallthinker_moe_sparse] examined=%zu activated=%zu "
                 "(relu_zero=%.2f%%)\n",
                 ex, activated.load(),
                 100.0 * (double)(ex - activated.load()) / (double)ex);
    long long mx = 0, mn = -1, sum = 0;
    unsigned n = 0;
    for (unsigned t = 0; t < kMaxThreads; ++t) {
      long long v = busy_ns[t].load();
      if (v == 0)
        continue;
      mx = std::max(mx, v);
      mn = (mn < 0) ? v : std::min(mn, v);
      sum += v;
      ++n;
    }
    if (n > 1) {
      double mean = (double)sum / n;
      std::fprintf(stderr,
                   "[smallthinker_moe_sparse] up/down per-thread busy ms: "
                   "min=%.1f max=%.1f mean=%.1f skew=(max-min)/mean=%.1f%% "
                   "(threads=%u)\n",
                   mn / 1e6, mx / 1e6, mean / 1e6,
                   100.0 * (double)(mx - mn) / mean, n);
    }
  }
};

const bool kLogSparsity = (std::getenv("NNTR_RELU_SPARSITY_LOG") != nullptr);
const bool kProfileThreads =
  (std::getenv("NNTR_MOE_THREAD_PROFILE") != nullptr);
// Up/down work distribution: 0 (default) = B1a static even-split of the active
// list (balanced by count, no atomic overhead); >0 = B1b dynamic work-stealing
// with that chunk size (balances wall-clock on heterogeneous cores, but adds
// atomic contention — only wins when per-expert active work is large enough).
const unsigned kActiveChunk = []() {
  const char *e = std::getenv("NNTR_MOE_CHUNK");
  return e ? (unsigned)std::atoi(e) : 0u;
}();
SparsityStats g_stats;

} // namespace

// Shared sparse ReGLU FFN (B3 hybrid): used by the resident base-sparse layer
// and the cached-slim / slim sparse variants. Declared in
// smallthinker_sparse_ffn.h. Kernels + env handling are file-static above.
void sparse_reglu_ffn(
  const nntrainer::Tensor &input, nntrainer::Tensor &out,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  long long *active_out, long long *examined_out) {

  const unsigned num_tokens = token_assignments.size();
  if (num_tokens == 0)
    return;

  // Number of intermediate neurons. The weight tensor dim is [.., hidden,
  // intermediate] (width = intermediate); the raw bytes are neuron-major plain
  // Q4_0 [intermediate, hidden].
  const unsigned intermediate_size = gate_proj.width();
  const unsigned nb = (hidden_size + 31) / 32; // Q4_0/Q8_0 blocks per row
  const size_t row_stride = kQ4_0_BLOCK * nb;

  // B3 hybrid: gate is now stored REPACKED and computed in full by the optimal
  // dense q4_0xq8_0 GEMV; only up/down are plain per-neuron Q4_0.
  const uint8_t *up_w = up_proj.getData<uint8_t>();
  const uint8_t *down_w = down_proj.getData<uint8_t>();

  float *out_data = out.getData<float>();

  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned compute_threads = tm.getComputeThreadCount();
  const unsigned thread_num = compute_threads == 0 ? 1 : compute_threads;

  // Per-thread output accumulation buffers (PowerInfer local_buf), reused
  // across tokens. Reduced into `out` once per token with the routing weight.
  std::vector<std::vector<float>> y_local(
    thread_num, std::vector<float>(hidden_size, 0.0f));
  // Activation quantized to Q8_0 once per token for the plain up dots.
  std::vector<uint8_t> xq(kQ8_0_BLOCK * nb);
  // Active (post-ReLU) neuron indices, compacted per token.
  std::vector<unsigned> active;
  active.reserve(intermediate_size);
  // Shared cursor for B1b dynamic work-stealing (reset per token).
  std::atomic<unsigned> next_active{0};

  const nntrainer::TensorDim token_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  const nntrainer::TensorDim gate_dim({1, 1, 1, intermediate_size},
                                      input.getTensorType());

  size_t local_examined = 0, local_activated = 0;

  for (unsigned ti = 0; ti < num_tokens; ++ti) {
    const unsigned token_idx = token_assignments[ti].first;
    const float weight = token_assignments[ti].second;
    const size_t token_off = (size_t)token_idx * hidden_size;
    float *out_row = out_data + token_off;

    // --- Gate phase: full gate via the optimal repacked q4_0xq8_0 GEMV. ---
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_dim, token_off, true);
    nntrainer::Tensor gate_out(gate_dim);
    token_input.dot(gate_proj, gate_out);
    const float *g = gate_out.getData<float>();

    // ReLU mask -> compact active set (g[j] = ReLU(gate_j) when g[j] > 0).
    active.clear();
    for (unsigned j = 0; j < intermediate_size; ++j)
      if (g[j] > 0.0f)
        active.push_back(j);
    const unsigned n_active = (unsigned)active.size();

    // Quantize x->q8_0 once for the plain up dots.
    quantize_row_q8_0_local(input.getData<float>() + token_off, xq.data(),
                            hidden_size);

    for (unsigned t = 0; t < thread_num; ++t)
      std::fill(y_local[t].begin(), y_local[t].end(), 0.0f);

    // --- Up/Down phase: active neurons only, distributed across threads.
    // B1b: dynamic work-stealing (atomic chunk counter) so faster cores grab
    // more chunks — balances wall-clock on heterogeneous P/E cores where an
    // equal-count static split leaves the fast cores idling (~18% skew). One
    // barrier. ---
    next_active.store(0, std::memory_order_relaxed);
    tm.parallel_for(0, (size_t)thread_num, [&](size_t t) {
      const auto t0 = std::chrono::steady_clock::now();
      float *yl = y_local[t].data();
      const uint8_t *xq_p = xq.data();
      auto run_neuron = [&](unsigned idx) {
        const unsigned j = active[idx];
        const float u = q4_0_row_dot_q8(up_w + row_stride * j, xq_p, nb);
        const float h = g[j] * u;
        q4_0_row_axpy(h, down_w + row_stride * j, yl, nb);
      };
      if (kActiveChunk == 0) {
        // B1a: static even split (every element active -> balanced by count).
        const unsigned s = (unsigned)((t * n_active) / thread_num);
        const unsigned e = (unsigned)(((t + 1) * n_active) / thread_num);
        for (unsigned idx = s; idx < e; ++idx)
          run_neuron(idx);
      } else {
        // B1b: dynamic work-stealing.
        for (;;) {
          const unsigned beg =
            next_active.fetch_add(kActiveChunk, std::memory_order_relaxed);
          if (beg >= n_active)
            break;
          const unsigned e = std::min(n_active, beg + kActiveChunk);
          for (unsigned idx = beg; idx < e; ++idx)
            run_neuron(idx);
        }
      }
      if (kProfileThreads && t < kMaxThreads)
        g_stats.busy_ns[t].fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0)
            .count(),
          std::memory_order_relaxed);
    });

    // Reduce per-thread buffers into the output row, scaled by routing weight.
    for (unsigned t = 0; t < thread_num; ++t) {
      const float *yl = y_local[t].data();
      for (unsigned k = 0; k < hidden_size; ++k)
        out_row[k] += weight * yl[k];
    }

    local_examined += intermediate_size;
    local_activated += n_active;
  }

  if (active_out)
    *active_out += (long long)local_activated;
  if (examined_out)
    *examined_out += (long long)local_examined;
  if (kLogSparsity) {
    g_stats.examined.fetch_add(local_examined, std::memory_order_relaxed);
    g_stats.activated.fetch_add(local_activated, std::memory_order_relaxed);
  }
}

void SmallThinkerSparseMoELayer::compute_sparse_ffn(
  const nntrainer::Tensor &input, nntrainer::Tensor &out,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {
  sparse_reglu_ffn(input, out, token_assignments, gate_proj, up_proj, down_proj,
                   hidden_size);
}

void SmallThinkerSparseMoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {
  // expert_output is a per-expert buffer (zero-initialized by the caller and
  // summed afterwards), so accumulate into it directly.
  compute_sparse_ffn(input, expert_output, token_assignments, gate_proj, up_proj,
                     down_proj, hidden_size);
}

void SmallThinkerSparseMoELayer::compute_expert_forward_batched(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {
  // Prefill: the plain per-neuron layout cannot use the repacked batched GEMM,
  // so run the same per-token sparse FFN. The caller drives the expert loop
  // serially, so accumulating into the shared `output` is race-free.
  compute_sparse_ffn(input, output, token_assignments, gate_proj, up_proj,
                     down_proj, hidden_size);
}

} // namespace causallm
