// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   q4_0_row_kernels.h
 * @date   30 June 2026
 * @brief  Plain (un-repacked) Q4_0 single-row kernels for sparse gather paths.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @note   These mirror the fused q4_0xq8_0 row kernels used by the SmallThinker
 *         sparse ReGLU FFN (smallthinker_moe_layer_base_sparse.cpp). They are
 *         duplicated here (header-inline) so the LM-head sparse predictor can
 *         reuse them WITHOUT modifying the audited FFN translation unit. The
 *         plain weight layout (per-row block_q4_0, NO ISA interleave) is
 *         produced by quantize_stream's writeMatrixPlain / *Plain helpers, so a
 *         single output row is addressable at runtime.
 */

#ifndef __Q4_0_ROW_KERNELS_H__
#define __Q4_0_ROW_KERNELS_H__
#ifdef __cplusplus

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace causallm {
namespace q4_0_row {

// Plain Q4_0 weight block: fp16 delta (2B) + 16 nibble bytes (32 weights) = 18B
// (matches the .bin layout produced by quantize_stream).
constexpr size_t kQ4_0_D = sizeof(uint16_t);
constexpr size_t kQ4_0_BLOCK = kQ4_0_D + 16;
// Activation Q8_0 block (produced AND consumed here, so the layout is ours):
// fp32 delta (4B) + 32 int8 quants = 36B.
constexpr size_t kQ8_0_D = sizeof(float);
constexpr size_t kQ8_0_BLOCK = kQ8_0_D + 32;

// Quantize one fp32 row to our Q8_0 activation layout, block by block.
static inline void quantize_row_q8_0_local(const float *x, uint8_t *q,
                                           unsigned n) {
  const unsigned nb = (n + 31) / 32;
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
      out = sign;
    } else {
      exp = 1;
      while ((man & 0x400) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      out = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000u | (man << 13);
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
    const __m128i bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q4 + kQ4_0_D));
    const __m128i lo = _mm_and_si128(bytes, low_mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
    __m256i qx = _mm256_set_m128i(hi, lo);
    qx = _mm256_sub_epi8(qx, off);
    const __m256i qy =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + kQ8_0_D));
    const __m256i ax = _mm256_sign_epi8(qx, qx);
    const __m256i sy = _mm256_sign_epi8(qy, qx);
    const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    vacc = _mm256_add_ps(
      vacc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32)));
  }
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
    const int8x16_t lo =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, low_mask)), off);
    const int8x16_t hi =
      vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), off);
    const int8x16_t q8lo =
      vld1q_s8(reinterpret_cast<const int8_t *>(q8 + kQ8_0_D));
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

} // namespace q4_0_row
} // namespace causallm

#endif /* __cplusplus */
#endif /* __Q4_0_ROW_KERNELS_H__ */
