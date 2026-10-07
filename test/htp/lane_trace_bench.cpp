// SPDX-License-Identifier: Apache-2.0
/** @brief Isolated synthetic MoE timeline and trace-on/off device check.
 * This is not an end-to-end model benchmark. Weights are distinct per expert;
 * every measured call uses identical inputs and verifies bitwise output.
 */
#include "hexkl_lane_trace.h"
#include "nntr_hvx.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <remote.h>
#include <string>
#include <vector>

static void check(int rc, const char *what) {
  if (rc) {
    std::fprintf(stderr, "%s failed: 0x%x\n", what, rc);
    std::exit(1);
  }
}
static double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>((v.size() - 1) * q)];
}
int main(int argc, char **argv) {
  const std::string prefix = argc > 1 ? argv[1] : "lane";
  const unsigned reps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20;
  const bool timed = argc > 3 && std::string(argv[3]) == "timed";
  if (!reps || reps > 1000)
    return 2;
  remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof(ud)),
        "unsigned PD");
  remote_handle64 s = 0;
  check(nntr_hvx_open((std::string(nntr_hvx_URI) + "&_dom=cdsp").c_str(), &s),
        "open");
  constexpr unsigned K = 2048, I = 1792, E = 32, TOP = 4;
  std::vector<uint32_t> gu(E), dn(E);
  for (unsigned side = 0; side < 2; ++side) {
    const unsigned wk = side ? I : K, wn = side ? K : 2 * I;
    std::vector<int8_t> w(wk * wn);
    std::vector<float> scales(wn, 0.015625f), bias(wn, 0.f);
    std::vector<int32_t> sums(wn);
    for (unsigned e = 0; e < E; ++e) {
      std::fill(sums.begin(), sums.end(), 0);
      uint32_t seed = 17 + e * 101 + side;
      for (unsigned k = 0; k < wk; ++k)
        for (unsigned n = 0; n < wn; ++n) {
          seed = seed * 1664525u + 1013904223u;
          const int8_t v = static_cast<int8_t>((seed >> 28) - 8);
          w[k * wn + n] = v;
          sums[n] += v;
        }
      check(nntr_hvx_weight_register_u8i4(
              s, wk, wn, w.data(), w.size(), scales.data(), wn, sums.data(), wn,
              bias.data(), wn, &(side ? dn : gu)[e]),
            "register");
    }
  }
  for (const unsigned M : {1u, 512u}) {
    std::vector<uint32_t> rows, counts(E);
    for (unsigned e = 0; e < E; ++e)
      for (unsigned m = 0; m < M; ++m)
        for (unsigned t = 0; t < TOP; ++t)
          if ((m + t) % E == e) {
            rows.push_back(m);
            ++counts[e];
          }
    std::vector<float> rw(rows.size(), 0.25f), act(M * K), out(M * K), ref;
    for (unsigned j = 0; j < act.size(); ++j)
      act[j] = static_cast<float>(static_cast<int>(j * 17 % 127) - 63) / 64;
    uint32_t stages[19] = {}; // MOE_N_STAGES in nntr_hvx_mm_u8i4.c
    auto invoke = [&]() {
      const auto a = std::chrono::steady_clock::now();
      if (timed)
        check(nntr_hvx_mm_u8i4_moe_layer_timed(
                s, M, K, I, K, gu.data(), E, dn.data(), E, rows.data(),
                rows.size(), counts.data(), E, rw.data(), rw.size(), act.data(),
                act.size(), out.data(), out.size(), stages, 19),
              "moe timed");
      else
        check(nntr_hvx_mm_u8i4_moe_layer(
                s, M, K, I, K, gu.data(), E, dn.data(), E, rows.data(),
                rows.size(), counts.data(), E, rw.data(), rw.size(), act.data(),
                act.size(), out.data(), out.size()),
              "moe");
      return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - a)
        .count();
    };
    for (unsigned warm = 0; warm < 5; ++warm)
      invoke();
    ref = out;
    std::vector<double> off, on;
    std::vector<double> dsp_off, dsp_on;
    std::vector<uint32_t> words(HEXKL_LANE_TRACE_MAX_WORDS);
    uint32_t used = 0;
    // Alternate order to reduce thermal/order bias. Control and read RPCs
    // are outside the measured call; enabled ring reset is inside it.
    for (unsigned r = 0; r < reps; ++r)
      for (unsigned j = 0; j < 2; ++j) {
        const bool enabled = ((r + j) & 1u) != 0;
        check(nntr_hvx_lane_trace_control(s, enabled), "trace control");
        (enabled ? on : off).push_back(invoke());
        if (timed)
          (enabled ? dsp_on : dsp_off).push_back(stages[0]);
        if (std::memcmp(out.data(), ref.data(), out.size() * sizeof(float))) {
          std::fprintf(stderr, "output changed M=%u trace=%d\n", M, enabled);
          return 1;
        }
        for (float v : out)
          if (!std::isfinite(v))
            return 1;
        if (enabled)
          check(nntr_hvx_lane_trace_read(s, words.data(), words.size(), &used),
                "trace read");
      }
    const std::string path = prefix + "-m" + std::to_string(M) + ".bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(words.data(), sizeof(uint32_t), used, f) != used)
      return 1;
    std::fclose(f);
    std::printf("M=%u reps=%u off_p50_us=%.3f on_p50_us=%.3f "
                "off_p95_us=%.3f on_p95_us=%.3f overhead_pct=%.2f "
                "records=%u dropped=%u entry_us=%.3f exact_output=yes\n",
                M, reps, percentile(off, .5), percentile(on, .5),
                percentile(off, .95), percentile(on, .95),
                (percentile(on, .5) / percentile(off, .5) - 1) * 100, words[3],
                words[4], words[5] / 19.2);
    for (unsigned r = 0; r < reps; ++r)
      std::printf("sample M=%u pair=%u off_us=%.3f on_us=%.3f\n", M, r, off[r],
                  on[r]);
    if (timed)
      std::printf("DSP_TIMED M=%u off_p50_us=%.3f on_p50_us=%.3f "
                  "overhead_pct=%.2f (stage probes enabled on both sides)\n",
                  M, percentile(dsp_off, .5), percentile(dsp_on, .5),
                  (percentile(dsp_on, .5) / percentile(dsp_off, .5) - 1) * 100);
    std::fflush(stdout);
  }
  for (unsigned e = 0; e < E; ++e) {
    check(nntr_hvx_weight_release_u8i4(s, gu[e]), "release gu");
    check(nntr_hvx_weight_release_u8i4(s, dn[e]), "release dn");
  }
  check(nntr_hvx_close(s), "close");
}
