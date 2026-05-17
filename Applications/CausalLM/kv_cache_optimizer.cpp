// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   kv_cache_optimizer.cpp
 * @date   17 May 2026
 * @brief  KV cache optimizer implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_optimizer.h"

#include <stdexcept>

namespace causallm {

void RawKVCacheOptimizer::allocate(const KVCacheSpec &spec) {
  batch_size_ = spec.batch_size;
  max_seq_len_ = spec.max_seq_len;
  num_heads_kv_ = spec.num_heads_kv;
  head_dim_ = spec.head_dim;
  kv_width_ = spec.num_heads_kv * spec.head_dim;
  dtype_ = spec.dtype;
  format_ = spec.format;

  ml::train::TensorDim cache_dim({batch_size_, 1, max_seq_len_, kv_width_},
                                 {format_, dtype_});

  layer_caches_.resize(spec.num_layers);
  for (unsigned int i = 0; i < spec.num_layers; ++i) {
    layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].key_cache.setZero();
    layer_caches_[i].value_cache.setZero();
  }
}

nntrainer::Tensor &RawKVCacheOptimizer::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheOptimizer::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].key_cache;
}

nntrainer::Tensor &RawKVCacheOptimizer::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheOptimizer::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].value_cache;
}

nntrainer::Tensor
RawKVCacheOptimizer::getKeyCacheWriteView(unsigned int layer_idx,
                                          unsigned int batch,
                                          unsigned int cache_pos,
                                          unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheOptimizer::getKeyCacheWriteView: invalid layer_idx");
  }
  if (cache_pos + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheOptimizer::getKeyCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width_},
                                {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos * kv_width_;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor
RawKVCacheOptimizer::getValueCacheWriteView(unsigned int layer_idx,
                                            unsigned int batch,
                                            unsigned int cache_pos,
                                            unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheOptimizer::getValueCacheWriteView: invalid layer_idx");
  }
  if (cache_pos + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheOptimizer::getValueCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width_},
                                {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos * kv_width_;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor
RawKVCacheOptimizer::getKeyCacheReadView(unsigned int layer_idx,
                                         unsigned int batch,
                                         unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheOptimizer::getKeyCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheOptimizer::getKeyCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width_}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

nntrainer::Tensor
RawKVCacheOptimizer::getValueCacheReadView(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheOptimizer::getValueCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheOptimizer::getValueCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width_}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

void RawKVCacheOptimizer::save(std::ostream &out,
                               unsigned int seq_len) const {
  for (const auto &lc : layer_caches_) {
    ml::train::TensorDim save_dim = lc.key_cache.getDim();
    save_dim.height(seq_len);

    nntrainer::Tensor k_slice = const_cast<nntrainer::Tensor &>(lc.key_cache)
                                  .getSharedDataTensor(save_dim, 0, true);
    nntrainer::Tensor v_slice = const_cast<nntrainer::Tensor &>(lc.value_cache)
                                  .getSharedDataTensor(save_dim, 0, true);

    k_slice.save(out);
    v_slice.save(out);
  }
}

void RawKVCacheOptimizer::load(std::ifstream &in, unsigned int seq_len) {
  for (auto &lc : layer_caches_) {
    ml::train::TensorDim load_dim = lc.key_cache.getDim();
    load_dim.height(seq_len);

    nntrainer::Tensor k_slice =
      lc.key_cache.getSharedDataTensor(load_dim, 0, true);
    nntrainer::Tensor v_slice =
      lc.value_cache.getSharedDataTensor(load_dim, 0, true);

    k_slice.read(in);
    v_slice.read(in);
  }
}

} // namespace causallm
