// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file moe_profiler.h
 * @brief Lightweight phase profiler for CausalLM MoE layers
 */

#ifndef __CAUSALLM_MOE_PROFILER_H__
#define __CAUSALLM_MOE_PROFILER_H__

#include <chrono>
#include <string>
#include <utility>

#ifdef PROFILE
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

namespace causallm {

/**
 * @brief Accumulates MoE phase durations without affecting normal inference.
 *
 * Build with enable-profile=true and set NNTR_MOE_PROFILE to a positive
 * integer N to print one aggregated report for every N layer invocations.
 * Timings inside expert worker threads are summed work time, while total is
 * elapsed wall time.
 */
class MoEProfiler {
public:
  enum class Phase {
    ROUTER,
    DISPATCH,
    MMAP,
    EXPERT,
    GATE_UP,
    ACTIVATION,
    DOWN,
    REDUCE,
    TOTAL,
    COUNT
  };

  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

#ifdef PROFILE
  MoEProfiler() : calls(0), tokens(0) {
    for (auto &duration : durations)
      duration.store(0, std::memory_order_relaxed);
  }

  MoEProfiler(MoEProfiler &&rhs) noexcept :
    name(std::move(rhs.name)),
    calls(rhs.calls.load(std::memory_order_relaxed)),
    tokens(rhs.tokens.load(std::memory_order_relaxed)) {
    for (size_t i = 0; i < durations.size(); ++i)
      durations[i].store(rhs.durations[i].load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  }

  MoEProfiler &operator=(MoEProfiler &&rhs) noexcept {
    if (this == &rhs)
      return *this;

    name = std::move(rhs.name);
    calls.store(rhs.calls.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    tokens.store(rhs.tokens.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    for (size_t i = 0; i < durations.size(); ++i)
      durations[i].store(rhs.durations[i].load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    return *this;
  }

  void setName(const std::string &layer_name) { name = layer_name; }

  bool enabled() const { return reportInterval() != 0; }

  TimePoint start() const { return enabled() ? Clock::now() : TimePoint(); }

  void record(Phase phase, TimePoint start_time) {
    if (!enabled())
      return;

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           Clock::now() - start_time)
                           .count();
    durations[static_cast<size_t>(phase)].fetch_add(
      static_cast<uint64_t>(elapsed), std::memory_order_relaxed);
  }

  void finish(unsigned int token_count) {
    const unsigned int interval = reportInterval();
    if (interval == 0)
      return;

    tokens.fetch_add(token_count, std::memory_order_relaxed);
    const uint64_t invocation =
      calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (invocation % interval != 0)
      return;

    std::lock_guard<std::mutex> lock(report_mutex);
    std::array<uint64_t, static_cast<size_t>(Phase::COUNT)> snapshot{};
    for (size_t i = 0; i < snapshot.size(); ++i)
      snapshot[i] = durations[i].exchange(0, std::memory_order_relaxed);

    const uint64_t reported_tokens =
      tokens.exchange(0, std::memory_order_relaxed);
    const auto us = [&snapshot](Phase phase) {
      return snapshot[static_cast<size_t>(phase)] / 1000ULL;
    };

    std::fprintf(
      stderr,
      "[MoEProfile] %s calls=%u tokens=%llu total_wall_us=%llu "
      "router_work_us=%llu dispatch_work_us=%llu mmap_work_us=%llu "
      "expert_wall_us=%llu gate_up_work_us=%llu activation_work_us=%llu "
      "down_work_us=%llu reduce_work_us=%llu\n",
      name.c_str(), interval, static_cast<unsigned long long>(reported_tokens),
      static_cast<unsigned long long>(us(Phase::TOTAL)),
      static_cast<unsigned long long>(us(Phase::ROUTER)),
      static_cast<unsigned long long>(us(Phase::DISPATCH)),
      static_cast<unsigned long long>(us(Phase::MMAP)),
      static_cast<unsigned long long>(us(Phase::EXPERT)),
      static_cast<unsigned long long>(us(Phase::GATE_UP)),
      static_cast<unsigned long long>(us(Phase::ACTIVATION)),
      static_cast<unsigned long long>(us(Phase::DOWN)),
      static_cast<unsigned long long>(us(Phase::REDUCE)));
  }
#else
  MoEProfiler() = default;
  MoEProfiler(MoEProfiler &&) noexcept = default;
  MoEProfiler &operator=(MoEProfiler &&) noexcept = default;

  void setName(const std::string &) {}

  bool enabled() const { return false; }

  TimePoint start() const { return TimePoint(); }

  void record(Phase, TimePoint) {}

  void finish(unsigned int) {}
#endif

private:
#ifdef PROFILE
  static unsigned int reportInterval() {
    static const unsigned int interval = []() {
      const char *value = std::getenv("NNTR_MOE_PROFILE");
      if (value == nullptr)
        return 0U;

      char *end = nullptr;
      const unsigned long parsed = std::strtoul(value, &end, 10);
      if (end == value || *end != '\0' || parsed == 0)
        return 0U;
      return static_cast<unsigned int>(parsed);
    }();
    return interval;
  }

  std::string name;
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Phase::COUNT)>
    durations;
  std::atomic<uint64_t> calls;
  std::atomic<uint64_t> tokens;
  std::mutex report_mutex;
#endif
};

} // namespace causallm

#endif // __CAUSALLM_MOE_PROFILER_H__
