// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen_moe_cache_policy.h
 * @date   29 July 2026
 * @brief  Cache planning helpers for the cached Qwen3 MoE layer
 * @see    https://github.com/nnstreamer/nntrainer
 */

#ifndef __QWEN_MOE_CACHE_POLICY_H__
#define __QWEN_MOE_CACHE_POLICY_H__

#include <cstddef>
#include <deque>
#include <list>
#include <unordered_map>
#include <vector>

namespace causallm::qwen3_cached_slim_detail {

/**
 * @brief Plan the post-forward expert cache in least-to-most-recent order
 *
 * This mirrors the existing cache update policy without requiring all active
 * experts to be mapped first.
 */
inline std::list<int> planExpertCache(const std::list<int> &loaded_experts,
                                      const std::vector<int> &active_experts,
                                      const std::deque<int> &expert_recency,
                                      std::size_t cache_capacity) {
  std::list<int> planned_cache = loaded_experts;
  std::unordered_map<int, std::list<int>::iterator> positions;
  positions.reserve(loaded_experts.size() + active_experts.size());

  for (auto iter = planned_cache.begin(); iter != planned_cache.end(); ++iter)
    positions.emplace(*iter, iter);

  for (int expert_idx : active_experts) {
    if (positions.find(expert_idx) != positions.end())
      continue;

    planned_cache.push_back(expert_idx);
    positions.emplace(expert_idx, --planned_cache.end());
  }

  for (auto iter = expert_recency.rbegin(); iter != expert_recency.rend();
       ++iter) {
    auto position = positions.find(*iter);
    if (position == positions.end())
      continue;

    planned_cache.splice(planned_cache.end(), planned_cache, position->second);
  }

  while (planned_cache.size() > cache_capacity) {
    positions.erase(planned_cache.front());
    planned_cache.pop_front();
  }

  return planned_cache;
}

} // namespace causallm::qwen3_cached_slim_detail

#endif /* __QWEN_MOE_CACHE_POLICY_H__ */
