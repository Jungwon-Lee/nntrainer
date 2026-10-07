// SPDX-License-Identifier: Apache-2.0
/** @file hexkl_lane_trace.h
 * @brief Bounded, one-call DSP timeline. Times are raw qtimer ticks.
 *
 * Each physical CPU thread has its own ring. A pool callback's slice index
 * is not a thread id (submit shifts it, and background units are dynamic).
 * HMX spans measure the caller issuing a batch, not hardware utilization.
 * DMA records describe outstanding descriptors with completion bounds, not
 * a hardware timestamp. Keep these distinctions in consumers.
 */
#ifndef __HEXKL_LANE_TRACE_H__
#define __HEXKL_LANE_TRACE_H__
#include <stdint.h>

#define HEXKL_LANE_TRACE_MAGIC 0x4e4c5431u
#define HEXKL_LANE_TRACE_THREADS 8u
#define HEXKL_LANE_TRACE_CAP 4096u
#define HEXKL_LANE_TRACE_HEADER_WORDS 8u
#define HEXKL_LANE_TRACE_RECORD_WORDS 8u
#define HEXKL_LANE_TRACE_MAX_WORDS                                             \
  (HEXKL_LANE_TRACE_HEADER_WORDS +                                             \
   HEXKL_LANE_TRACE_THREADS * HEXKL_LANE_TRACE_CAP * 8u)

enum {
  HLT_HMX_GU = 1,
  HLT_HMX_DN,
  HLT_WORKER,
  HLT_BG_UNIT,
  HLT_WAIT_FG,
  HLT_WAIT_BG,
  HLT_DMA_PENDING,
  HLT_WAIT_DMA,
  HLT_DMA_PUSH,
  HLT_SUBMIT,
  HLT_QUANT,
  HLT_PACK,
  HLT_GU_EPILOGUE,
  HLT_DN_EPILOGUE,
  HLT_TAIL_GU,
  HLT_TAIL_RQ,
  HLT_TAIL_DN,
  HLT_HMX_CONV_AC,
  HLT_HMX_CONV_B,
  HLT_HMX_CONV_OUT,
  HLT_CONV_AC_EPILOGUE,
  HLT_CONV_EPILOGUE,
  HLT_CONV_STAGE
};

#ifdef NNTR_DSP_LANE_TRACE
extern int hexkl_lane_trace_on;
uint64_t hexkl_lane_trace_now(void);
void hexkl_lane_trace_arm(int enable);
void hexkl_lane_trace_begin(void);
void hexkl_lane_trace_end(void);
void hexkl_lane_trace_record(uint32_t writer, uint32_t kind, uint64_t t0,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             uint32_t a3);
void hexkl_lane_trace_func(uintptr_t func, uint32_t kind);
uint32_t hexkl_lane_trace_kind(uintptr_t func, uint32_t fallback);
uint32_t hexkl_lane_trace_job(void);
void hexkl_lane_trace_dma_sample(void);
void hexkl_lane_trace_dma_issue(uint32_t slot, uint32_t bytes, uint64_t t0);
uint32_t hexkl_lane_trace_dma_id(uint32_t slot);
uint32_t hexkl_lane_trace_copy(uint32_t *words, uint32_t cap);
#else
#define hexkl_lane_trace_on 0
static inline void hexkl_lane_trace_arm(int enable) { (void)enable; }
static inline void hexkl_lane_trace_begin(void) {}
static inline void hexkl_lane_trace_end(void) {}
static inline uint32_t hexkl_lane_trace_copy(uint32_t *words, uint32_t cap) {
  (void)words;
  (void)cap;
  return 0;
}
static inline uint64_t hexkl_lane_trace_now(void) { return 0; }
static inline void hexkl_lane_trace_dma_sample(void) {}
static inline void hexkl_lane_trace_func(uintptr_t f, uint32_t k) {
  (void)f;
  (void)k;
}
static inline uint32_t hexkl_lane_trace_kind(uintptr_t f, uint32_t k) {
  (void)f;
  return k;
}
static inline uint32_t hexkl_lane_trace_job(void) { return 0; }
static inline uint32_t hexkl_lane_trace_dma_id(uint32_t s) {
  (void)s;
  return 0;
}
static inline void hexkl_lane_trace_dma_issue(uint32_t s, uint32_t b,
                                              uint64_t t) {
  (void)s;
  (void)b;
  (void)t;
}
static inline void hexkl_lane_trace_record(uint32_t w, uint32_t k, uint64_t t,
                                           uint32_t a, uint32_t b, uint32_t c,
                                           uint32_t d) {
  (void)w;
  (void)k;
  (void)t;
  (void)a;
  (void)b;
  (void)c;
  (void)d;
}
#endif

#define HLT_BEGIN(t)                                                           \
  uint64_t t = hexkl_lane_trace_on ? hexkl_lane_trace_now() : 0
#define HLT_END(w, k, t, a, b, c, d)                                           \
  do {                                                                         \
    if (hexkl_lane_trace_on)                                                   \
      hexkl_lane_trace_record(w, k, t, a, b, c, d);                            \
  } while (0)
#endif
