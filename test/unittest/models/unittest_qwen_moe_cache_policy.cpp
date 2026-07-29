// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_qwen_moe_cache_policy.cpp
 * @date   29 July 2026
 * @brief  Qwen3 CachedSlim MoE cache policy unit tests
 * @see    https://github.com/nnstreamer/nntrainer
 */

#include <qwen_moe_cache_policy.h>

#include <gtest/gtest.h>

namespace {

TEST(Qwen3CachedSlimMoERoutingTest, CollectsExpertsFromRecentTokensFirst) {
  const std::vector<std::uint32_t> indices{
    0, 1, // oldest token
    2, 3, //
    4, 5, // most recent token
  };

  const auto recent = causallm::qwen_moe::detail::collectRecentExperts(
    indices.data(), 3, 2, 6, 4);

  EXPECT_EQ(recent, (std::vector<unsigned int>{4, 5, 2, 3}));
}

TEST(Qwen3CachedSlimMoERoutingTest, DeduplicatesExpertsBeforeCapacity) {
  const std::vector<std::uint32_t> indices{
    0, 1, //
    1, 2, //
    2, 1, //
  };

  const auto recent = causallm::qwen_moe::detail::collectRecentExperts(
    indices.data(), 3, 2, 3, 3);

  EXPECT_EQ(recent, (std::vector<unsigned int>{2, 1, 0}));
}

TEST(Qwen3CachedSlimMoERoutingTest, ReturnsAllRoutedExpertsBelowCapacity) {
  const std::vector<std::uint32_t> indices{1, 3, 1, 3};

  const auto recent = causallm::qwen_moe::detail::collectRecentExperts(
    indices.data(), 2, 2, 5, 32);

  EXPECT_EQ(recent, (std::vector<unsigned int>{1, 3}));
}

} // namespace
