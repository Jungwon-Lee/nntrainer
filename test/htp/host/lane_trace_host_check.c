// SPDX-License-Identifier: Apache-2.0
#include "hexkl_lane_trace.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t tick;
static int done[256];
uint64_t hexkl_lane_trace_now(void) { return tick; }
int hexkl_dma_ring_is_done(uint32_t slot) { return done[slot]; }

int main(void) {
  uint32_t *words = malloc(HEXKL_LANE_TRACE_MAX_WORDS * sizeof(uint32_t));
  assert(words);
  assert(!hexkl_lane_trace_copy(words, HEXKL_LANE_TRACE_MAX_WORDS));
  hexkl_lane_trace_begin();
  assert(!hexkl_lane_trace_on);
  tick = 100;
  hexkl_lane_trace_arm(1);
  hexkl_lane_trace_begin();
  assert(hexkl_lane_trace_on);
  assert(!hexkl_lane_trace_copy(words, HEXKL_LANE_TRACE_MAX_WORDS));
  tick = 120;
  hexkl_lane_trace_record(2, HLT_QUANT, 110, 1, 2, 3, 4);
  hexkl_lane_trace_dma_issue(255, 4096, tick);
  tick = 130;
  hexkl_lane_trace_dma_sample();
  hexkl_lane_trace_dma_issue(0, 8192, tick);
  done[255] = done[0] = 1;
  tick = 140;
  hexkl_lane_trace_dma_sample();
  tick = 150;
  hexkl_lane_trace_end();
  assert(hexkl_lane_trace_copy(words, HEXKL_LANE_TRACE_MAX_WORDS) == 32);
  assert(words[3] == 3 && words[4] == 0 && words[5] == 50);
  assert(words[8] == HLT_DMA_PENDING && words[8 + 6] == 30);
  assert(words[24] == HLT_QUANT && words[25] == 2 && words[26] == 10 &&
         words[27] == 10);
  assert(!hexkl_lane_trace_copy(words, 31));
  hexkl_lane_trace_begin(); /* arm is consumed once */
  assert(!hexkl_lane_trace_on);
  hexkl_lane_trace_arm(1);
  hexkl_lane_trace_begin();
  for (uint32_t i = 0; i < HEXKL_LANE_TRACE_CAP + 2; ++i)
    hexkl_lane_trace_record(0, HLT_WORKER, tick, 0, 0, 0, 0);
  hexkl_lane_trace_record(8, HLT_WORKER, tick, 0, 0, 0, 0);
  done[1] = 0;
  hexkl_lane_trace_dma_issue(1, 1, tick);
  hexkl_lane_trace_end();
  assert(hexkl_lane_trace_copy(words, HEXKL_LANE_TRACE_MAX_WORDS));
  assert(words[3] == HEXKL_LANE_TRACE_CAP && words[4] == 4);
  free(words);
  puts("LANE TRACE BOUNDS / RESET / OVERFLOW OK");
  return 0;
}
