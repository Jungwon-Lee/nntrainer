// SPDX-License-Identifier: Apache-2.0
/** @brief Synthetic conv-block output/state and DSP timeline device check.
 * Saved output/state buffers allow comparison of two skels built from the
 * same sources except for the conv kernel. This is not a model benchmark.
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

static void save(const std::string &path, const void *data, size_t bytes) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::perror(path.c_str());
    std::exit(1);
  }
  const bool written = std::fwrite(data, 1, bytes, f) == bytes;
  const int rc = std::fclose(f);
  if (!written || rc)
    std::exit(1);
}

static double p50(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[(values.size() - 1) / 2];
}

int main(int argc, char **argv) {
  const std::string prefix = argc > 1 ? argv[1] : "conv";
  const unsigned reps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20;
  const bool capture = argc > 3 && std::string(argv[3]) == "lane";
  if (!reps || reps > 1000)
    return 2;
  remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof(ud)),
        "unsigned PD");
  remote_handle64 session = 0;
  check(
    nntr_hvx_open((std::string(nntr_hvx_URI) + "&_dom=cdsp").c_str(), &session),
    "open");
  constexpr unsigned K = 2048, C = 2048, N = 2048;
  uint32_t handles[4] = {};
  for (unsigned side = 0; side < 4; ++side) {
    std::vector<int8_t> weight(K * C);
    std::vector<float> scales(C, 1.f / 128.f), bias(C, 0.f);
    std::vector<int32_t> sums(C);
    uint32_t seed = 17u + side * 101u;
    for (unsigned k = 0; k < K; ++k)
      for (unsigned c = 0; c < C; ++c) {
        seed = seed * 1664525u + 1013904223u;
        const int8_t v = static_cast<int8_t>(static_cast<int>(seed >> 28) - 8);
        weight[k * C + c] = v;
        sums[c] += v;
      }
    check(nntr_hvx_weight_register_u8i4(
            session, K, C, weight.data(), weight.size(), scales.data(), C,
            sums.data(), C, bias.data(), C, &handles[side]),
          "register conv weight");
  }
  std::vector<float> conv_w(3u * C);
  for (unsigned j = 0; j < conv_w.size(); ++j)
    conv_w[j] = static_cast<float>(static_cast<int>(j % 7u) - 3) / 16.f;
  for (const unsigned M : {1u, 3u, 63u, 65u, 150u, 512u}) {
    std::vector<float> act(M * K), out(M * N), state(2u * C);
    for (unsigned j = 0; j < act.size(); ++j)
      act[j] = static_cast<float>(static_cast<int>(j * 17u % 127u) - 63) / 64.f;
    uint32_t stages[19] = {}; // MOE_N_STAGES in nntr_hvx_mm_u8i4.c
    auto invoke = [&]() {
      const auto start = std::chrono::steady_clock::now();
      check(nntr_hvx_mm_u8i4_conv_block_timed(
              session, M, K, C, N, handles, 3, handles[3], conv_w.data(),
              conv_w.size(), act.data(), act.size(), out.data(), out.size(),
              state.data(), state.size(), stages, 19),
            "conv timed");
      return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - start)
        .count();
    };
    for (unsigned warm = 0; warm < 3; ++warm)
      invoke();
    std::vector<float> reference = out;
    reference.insert(reference.end(), state.begin(), state.end());
    std::vector<double> host, dsp;
    for (unsigned r = 0; r < reps; ++r) {
      host.push_back(invoke());
      dsp.push_back(stages[0]);
      if (std::memcmp(out.data(), reference.data(),
                      out.size() * sizeof(float)) ||
          std::memcmp(state.data(), reference.data() + out.size(),
                      state.size() * sizeof(float))) {
        std::fprintf(stderr, "conv output/state changed M=%u iteration=%u\n", M,
                     r);
        return 1;
      }
    }
    for (float v : reference)
      if (!std::isfinite(v))
        return 1;
    const std::string base = prefix + "-m" + std::to_string(M);
    save(base + ".f32", reference.data(), reference.size() * sizeof(float));
    if (capture) {
      check(nntr_hvx_lane_trace_control(session, 1), "trace control");
      invoke();
      if (std::memcmp(out.data(), reference.data(),
                      out.size() * sizeof(float)) ||
          std::memcmp(state.data(), reference.data() + out.size(),
                      state.size() * sizeof(float)))
        return 1;
      std::vector<uint32_t> words(HEXKL_LANE_TRACE_MAX_WORDS);
      uint32_t used = 0;
      check(
        nntr_hvx_lane_trace_read(session, words.data(), words.size(), &used),
        "trace read");
      if (used < HEXKL_LANE_TRACE_HEADER_WORDS || used > words.size() ||
          words[4])
        return 1;
      save(base + ".bin", words.data(), used * sizeof(uint32_t));
    }
    std::printf("M=%u reps=%u host_p50_us=%.3f dsp_p50_us=%.3f "
                "exact_output_and_state=yes\n",
                M, reps, p50(host), p50(dsp));
    for (unsigned r = 0; r < reps; ++r)
      std::printf("sample M=%u iteration=%u host_us=%.3f dsp_us=%.3f\n", M, r,
                  host[r], dsp[r]);
    std::fflush(stdout);
  }
  for (uint32_t handle : handles)
    check(nntr_hvx_weight_release_u8i4(session, handle), "release conv weight");
  check(nntr_hvx_close(session), "close");
}
