// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   nntr_ggml_impl.h
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#ifndef __NNTR_GGML_IMPL__
#define __NNTR_GGML_IMPL__

#include <stddef.h>
#include <stdint.h>

void nntr_ggml_init();

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

#ifdef ENABLE_FP16
// Pick the half type the same way tensor_dim.h does, so this header stays
// self-contained even if a caller pulls it in without tensor_dim.h. On
// ARM/Android (USE__FP16) it is __fp16; on x86_64 fp16 builds it is _Float16.
#ifdef USE__FP16
#define NNTR_GGML_FP16 __fp16
#else
#define NNTR_GGML_FP16 _Float16
#endif
void nntr_gemm_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);
#endif

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0_sparse(int n, float *__restrict s, size_t bs,
                                    const void *__restrict vx,
                                    const void *__restrict vy,
                                    const uint8_t *__restrict block_masks,
                                    int nr, int nc);

bool nntr_gemv_q4_0_4x8_q8_0_sparse_supported();

void nntr_gemv_q4_0_4x8_q8_0_output_masked(
  int n, float *__restrict s, size_t bs, const void *__restrict vx,
  const void *__restrict vy, const uint8_t *__restrict output_mask, int nr,
  int nc);

/**
 * @brief Decide whether an output-masked Q4_0x4 GEMV is likely beneficial.
 *
 * The ARM kernel skips a four-output tile with no active lanes and uses a
 * single-lane path for one active lane. Two or more active lanes use the dense
 * SIMD path. The weights below conservatively model the measured relative cost
 * of those paths and reject masks whose tile layout would only add dispatch
 * overhead.
 */
static inline bool
nntr_should_use_output_masked_q4_0(const uint8_t *output_mask, size_t size) {
  constexpr size_t tile_size = 4;
  constexpr size_t dense_cost = 20;
  constexpr size_t single_lane_cost = 18;
  constexpr size_t masked_dense_cost = 21;

  if (size % tile_size != 0)
    return false;

  size_t estimated_cost = 0;
  for (size_t offset = 0; offset < size; offset += tile_size) {
    unsigned int active_count = 0;
    for (size_t lane = 0; lane < tile_size; ++lane)
      active_count += output_mask[offset + lane] != 0;

    if (active_count == 1)
      estimated_cost += single_lane_cost;
    else if (active_count >= 2)
      estimated_cost += masked_dense_cost;
  }

  return estimated_cost < dense_cost * (size / tile_size);
}

bool nntr_gemv_q4_0_8x8_q8_0_masked_supported();

void nntr_gemv_q4_0_8x8_q8_0_output_masked(
  int n, float *__restrict s, size_t bs, const void *__restrict vx,
  const void *__restrict vy, const uint8_t *__restrict output_mask, int nr,
  int nc);

/**
 * @brief Decide whether an output-masked Q4_0x8 GEMV is likely beneficial.
 */
static inline bool
nntr_should_use_output_masked_q4_0_8x8(const uint8_t *output_mask,
                                       size_t size) {
  constexpr size_t tile_size = 8;
  constexpr size_t dense_cost = 20;
  constexpr size_t single_lane_cost = 18;
  constexpr size_t masked_dense_cost = 21;

  if (size % tile_size != 0)
    return false;

  size_t estimated_cost = 0;
  for (size_t offset = 0; offset < size; offset += tile_size) {
    unsigned int active_count = 0;
    for (size_t lane = 0; lane < tile_size; ++lane)
      active_count += output_mask[offset + lane] != 0;

    if (active_count == 1)
      estimated_cost += single_lane_cost;
    else if (active_count >= 2)
      estimated_cost += masked_dense_cost;
  }

  return estimated_cost < dense_cost * (size / tile_size);
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_quantize_mat_q8_0_4x4(const float *__restrict x, void *__restrict vy,
                                int64_t k);

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

int nntr_repack_q8_0_to_q8_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q8_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

void nntr_quantize_row_q8_0(const float *__restrict x, void *__restrict y,
                            int64_t k);

/**
 * @brief Build four 8-element non-zero masks for each Q8_0 block.
 *
 * @return true when at least one quarter of the sub-blocks can be skipped.
 */
bool nntr_build_q8_0_subblock_masks(const void *__restrict q8_data,
                                    int64_t num_blocks,
                                    uint8_t *__restrict block_masks);

void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k);

void nntr_dequantize_row_q4_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q4_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q6_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q8_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q8_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc);

#endif
