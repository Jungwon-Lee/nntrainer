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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace causallm::qwen_moe::detail {

constexpr std::size_t EXPERT_CACHE_CAPACITY = 32;

/**
 * @brief Collect the most recently routed unique experts
 *
 * @param topk_indices Flattened [total_tokens, topk] expert indices
 * @param total_tokens Number of routed tokens
 * @param topk Number of selected experts per token
 * @param num_experts Total number of experts
 * @param capacity Maximum number of experts to collect
 * @return Expert indices ordered from most to least recent
 */
inline std::vector<unsigned int>
collectRecentExperts(const std::uint32_t *topk_indices,
                     std::size_t total_tokens, std::size_t topk,
                     std::size_t num_experts, std::size_t capacity) {
  std::vector<unsigned int> recent_experts;
  recent_experts.reserve(std::min(capacity, num_experts));

  if (topk_indices == nullptr || total_tokens == 0 || topk == 0 ||
      num_experts == 0 || capacity == 0) {
    return recent_experts;
  }

  std::vector<bool> collected(num_experts, false);
  for (std::size_t token = total_tokens; token-- > 0;) {
    for (std::size_t rank = 0; rank < topk; ++rank) {
      const unsigned int expert_idx = topk_indices[token * topk + rank];
      if (expert_idx >= num_experts) {
        throw std::invalid_argument("Expert index exceeds number of experts");
      }

      if (collected[expert_idx])
        continue;

      collected[expert_idx] = true;
      recent_experts.push_back(expert_idx);
      if (recent_experts.size() == capacity)
        return recent_experts;
    }
  }

  return recent_experts;
}

/**
 * @brief Plan the post-forward expert cache in least-to-most-recent order
 *
 * This mirrors the existing cache update policy without requiring all active
 * experts to be mapped first.
 */
inline std::list<int> planExpertCache(const std::list<int> &loaded_experts,
                                      const std::vector<int> &active_experts,
                                      const std::vector<unsigned int>
                                        &recent_experts,
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

  for (auto iter = recent_experts.rbegin(); iter != recent_experts.rend();
       ++iter) {
    const int expert_idx = static_cast<int>(*iter);
    auto position = positions.find(expert_idx);
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

} // namespace causallm::qwen_moe::detail

#endif // __QWEN_MOE_CACHE_POLICY_H__
