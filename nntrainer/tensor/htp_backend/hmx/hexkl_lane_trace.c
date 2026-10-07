// SPDX-License-Identifier: Apache-2.0
/** @file hexkl_lane_trace.c
 * @brief One writer per ring; read/reset only after every pool job joins.
 */
#include "hexkl_lane_trace.h"
#include "hexkl_dma_ring.h"
#include <stdatomic.h>
#include <string.h>
#ifndef HEXKL_LANE_TRACE_HOST_TEST
#include <HAP_perf.h>
uint64_t hexkl_lane_trace_now(void) { return HAP_perf_get_qtimer_count(); }
#endif

typedef struct {
  uint32_t n, dropped;
  uint32_t rec[HEXKL_LANE_TRACE_CAP][HEXKL_LANE_TRACE_RECORD_WORDS];
} trace_ring;
static trace_ring rings[HEXKL_LANE_TRACE_THREADS];
static int armed, completed;
static _Atomic uint32_t invalid_writers;
int hexkl_lane_trace_on;
static uint64_t epoch, end_tick;
static uintptr_t funcs[16];
static uint32_t kinds[16], nfunc, job_seq, dma_seq;
typedef struct {
  uint64_t issue, last_not_done;
  uint32_t bytes, id;
  int pending;
} dma_rec;
static dma_rec dma[256];
static uint32_t dma_queue[256], dma_head, dma_count;

void hexkl_lane_trace_arm(int enable) { armed = enable != 0; }
void hexkl_lane_trace_begin(void) {
  if (!armed)
    return;
  armed = 0;
  completed = 0;
  for (uint32_t i = 0; i < HEXKL_LANE_TRACE_THREADS; ++i)
    rings[i].n = rings[i].dropped = 0;
  atomic_store_explicit(&invalid_writers, 0, memory_order_relaxed);
  memset(dma, 0, sizeof(dma));
  nfunc = job_seq = dma_seq = 0;
  dma_head = dma_count = 0;
  epoch = hexkl_lane_trace_now();
  end_tick = epoch;
  hexkl_lane_trace_on = 1;
}

void hexkl_lane_trace_record(uint32_t writer, uint32_t kind, uint64_t t0,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             uint32_t a3) {
  if (!hexkl_lane_trace_on)
    return;
  if (writer >= HEXKL_LANE_TRACE_THREADS) {
    atomic_fetch_add_explicit(&invalid_writers, 1, memory_order_relaxed);
    return;
  }
  const uint64_t now = hexkl_lane_trace_now();
  trace_ring *r = &rings[writer];
  if (r->n == HEXKL_LANE_TRACE_CAP) {
    ++r->dropped;
    return;
  }
  const uint32_t rec[8] = {
    kind, writer, (uint32_t)(t0 - epoch), (uint32_t)(now - t0), a0, a1, a2, a3};
  memcpy(r->rec[r->n++], rec, sizeof(rec));
}

void hexkl_lane_trace_func(uintptr_t f, uint32_t kind) {
  if (hexkl_lane_trace_on && nfunc < 16) {
    funcs[nfunc] = f;
    kinds[nfunc++] = kind;
  }
}
uint32_t hexkl_lane_trace_kind(uintptr_t f, uint32_t fallback) {
  if (hexkl_lane_trace_on)
    for (uint32_t i = 0; i < nfunc; ++i)
      if (funcs[i] == f)
        return kinds[i];
  return fallback;
}
uint32_t hexkl_lane_trace_job(void) {
  return hexkl_lane_trace_on ? ++job_seq : 0;
}

void hexkl_lane_trace_dma_sample(void) {
  if (!hexkl_lane_trace_on)
    return;
  const uint64_t now = hexkl_lane_trace_now();
  /* Descriptors complete in link order. Inspect only the outstanding head,
     not the whole ring, to keep polling overhead proportional to retirements.
   */
  while (dma_count) {
    const uint32_t i = dma_queue[dma_head];
    dma_rec *d = &dma[i];
    if (hexkl_dma_ring_is_done(i)) {
      hexkl_lane_trace_record(0, HLT_DMA_PENDING, d->issue, d->bytes, d->id,
                              (uint32_t)(d->last_not_done - epoch), i);
      d->pending = 0;
      dma_head = (dma_head + 1) & 255u;
      --dma_count;
    } else {
      d->last_not_done = now;
      break;
    }
  }
}
void hexkl_lane_trace_dma_issue(uint32_t slot, uint32_t bytes, uint64_t t0) {
  if (!hexkl_lane_trace_on)
    return;
  dma_rec *d = &dma[slot];
  d->issue = t0;
  d->last_not_done = t0;
  d->bytes = bytes;
  d->id = ++dma_seq;
  d->pending = 1;
  dma_queue[(dma_head + dma_count++) & 255u] = slot;
}
uint32_t hexkl_lane_trace_dma_id(uint32_t slot) {
  return hexkl_lane_trace_on ? dma[slot].id : 0;
}
void hexkl_lane_trace_end(void) {
  if (!hexkl_lane_trace_on)
    return;
  hexkl_lane_trace_dma_sample();
  end_tick = hexkl_lane_trace_now();
  hexkl_lane_trace_on = 0;
  completed = 1;
}
uint32_t hexkl_lane_trace_copy(uint32_t *words, uint32_t cap) {
  if (!completed || hexkl_lane_trace_on || cap < HEXKL_LANE_TRACE_HEADER_WORDS)
    return 0;
  uint32_t count = 0, dropped = 0;
  for (uint32_t i = 0; i < HEXKL_LANE_TRACE_THREADS; ++i) {
    count += rings[i].n;
    dropped += rings[i].dropped;
  }
  dropped += dma_count; /* An unfinished DMA makes this capture incomplete. */
  dropped += atomic_load_explicit(&invalid_writers, memory_order_relaxed);
  dropped += end_tick - epoch > UINT32_MAX;
  const uint32_t need = HEXKL_LANE_TRACE_HEADER_WORDS + count * 8u;
  if (cap < need)
    return 0;
  const uint32_t hdr[8] = {HEXKL_LANE_TRACE_MAGIC,
                           1u,
                           19200000u,
                           count,
                           dropped,
                           (uint32_t)(end_tick - epoch),
                           8u,
                           HEXKL_LANE_TRACE_THREADS};
  memcpy(words, hdr, sizeof(hdr));
  uint32_t *p = words + 8;
  for (uint32_t i = 0; i < HEXKL_LANE_TRACE_THREADS; ++i) {
    memcpy(p, rings[i].rec, rings[i].n * 32u);
    p += rings[i].n * 8u;
  }
  return need;
}
