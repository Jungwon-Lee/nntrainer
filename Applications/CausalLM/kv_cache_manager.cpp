// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  KV Cache Manager implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_manager.h"

#include <memory>
#include <stdexcept>

namespace causallm {

std::unique_ptr<KVCacheOptimizer>
KVCacheManager::selectOptimizer(const KVCacheSpec &spec) {
  if (spec.config.backend.empty() || spec.config.backend == "raw") {
    return std::make_unique<RawKVCacheOptimizer>();
  }

  if (spec.config.fallback == "raw") {
    return std::make_unique<RawKVCacheOptimizer>();
  }

  throw std::invalid_argument(
    "KVCacheManager::allocate: unsupported kv_cache backend: " +
    spec.config.backend);
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  KVCacheSpec spec;
  spec.num_layers = num_layers;
  spec.batch_size = batch_size;
  spec.max_seq_len = max_seq_len;
  spec.num_heads_kv = num_heads_kv;
  spec.head_dim = head_dim;
  spec.dtype = dtype;
  spec.format = format;

  allocate(spec);
}

void KVCacheManager::allocate(const KVCacheSpec &spec) {
  if (spec.num_layers == 0 || spec.batch_size == 0 ||
      spec.max_seq_len == 0 || spec.num_heads_kv == 0 ||
      spec.head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  auto optimizer = selectOptimizer(spec);
  optimizer->allocate(spec);

  num_layers_ = spec.num_layers;
  batch_size_ = spec.batch_size;
  max_seq_len_ = spec.max_seq_len;
  num_heads_kv_ = spec.num_heads_kv;
  head_dim_ = spec.head_dim;
  kv_width_ = spec.num_heads_kv * spec.head_dim;
  dtype_ = spec.dtype;
  format_ = spec.format;
  cache_pos_ = 0;
  optimizer_ = std::move(optimizer);
}

bool KVCacheManager::isAllocated() const {
  return optimizer_ != nullptr && optimizer_->isAllocated();
}

void KVCacheManager::setPosition(unsigned int pos) {
  if (pos > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::setPosition: pos exceeds max_seq_len");
  }
  cache_pos_ = pos;
}

void KVCacheManager::advance(unsigned int step_size) {
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::advance: position would exceed max_seq_len");
  }
  cache_pos_ += step_size;
}

void KVCacheManager::reset() { cache_pos_ = 0; }

nntrainer::Tensor &KVCacheManager::getKeyCache(unsigned int layer_idx) {
  if (!isAllocated()) {
    throw std::runtime_error("KVCacheManager::getKeyCache: not allocated");
  }
  return optimizer_->getKeyCache(layer_idx);
}

nntrainer::Tensor &KVCacheManager::getValueCache(unsigned int layer_idx) {
  if (!isAllocated()) {
    throw std::runtime_error("KVCacheManager::getValueCache: not allocated");
  }
  return optimizer_->getValueCache(layer_idx);
}

nntrainer::Tensor KVCacheManager::getKeyCacheWriteView(unsigned int layer_idx,
                                                       unsigned int batch,
                                                       unsigned int step_size) {
  if (!isAllocated())
    throw std::runtime_error(
      "KVCacheManager::getKeyCacheWriteView: not allocated");
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: would exceed max_seq_len");
  }

  return optimizer_->getKeyCacheWriteView(layer_idx, batch, cache_pos_,
                                          step_size);
}

nntrainer::Tensor KVCacheManager::getValueCacheWriteView(
  unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  if (!isAllocated())
    throw std::runtime_error(
      "KVCacheManager::getValueCacheWriteView: not allocated");
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: would exceed max_seq_len");
  }

  return optimizer_->getValueCacheWriteView(layer_idx, batch, cache_pos_,
                                            step_size);
}

nntrainer::Tensor KVCacheManager::getKeyCacheReadView(unsigned int layer_idx,
                                                      unsigned int batch,
                                                      unsigned int read_len) {
  if (!isAllocated())
    throw std::runtime_error(
      "KVCacheManager::getKeyCacheReadView: not allocated");
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: read_len exceeds max_seq_len");
  }

  return optimizer_->getKeyCacheReadView(layer_idx, batch, read_len);
}

nntrainer::Tensor KVCacheManager::getValueCacheReadView(unsigned int layer_idx,
                                                        unsigned int batch,
                                                        unsigned int read_len) {
  if (!isAllocated())
    throw std::runtime_error(
      "KVCacheManager::getValueCacheReadView: not allocated");
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: read_len exceeds max_seq_len");
  }

  return optimizer_->getValueCacheReadView(layer_idx, batch, read_len);
}

void KVCacheManager::save(const std::string &path) const {
  save(path, cache_pos_);
}

void KVCacheManager::save(const std::string &path, unsigned int seq_len) const {
  if (!isAllocated()) {
    throw std::runtime_error("KVCacheManager::save: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::save: seq_len exceeds max_seq_len");
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::save: cannot open file: " + path);
  }

  optimizer_->save(f, seq_len);
}

void KVCacheManager::load(const std::string &path, unsigned int seq_len) {
  if (!isAllocated()) {
    throw std::runtime_error("KVCacheManager::load: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::load: seq_len exceeds max_seq_len");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::load: cannot open file: " + path);
  }

  optimizer_->load(f, seq_len);
  cache_pos_ = seq_len;
}

} // namespace causallm
