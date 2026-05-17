// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   kv_cache_optimizer.h
 * @date   17 May 2026
 * @brief  KV cache optimizer abstraction
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __KV_CACHE_OPTIMIZER_H__
#define __KV_CACHE_OPTIMIZER_H__

#include <istream>
#include <ostream>
#include <fstream>
#include <string>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>

namespace causallm {

/**
 * @brief Runtime KV cache optimization policy parsed from nntr_config.json.
 */
struct KVCacheConfig {
  std::string backend = "raw";
  std::string format = "auto";
  std::string materialize_dtype = "auto";
  std::string scale_granularity = "per_token_per_head";
  std::string fallback = "error";
};

/**
 * @brief Complete allocation spec for a KV cache backend.
 */
struct KVCacheSpec {
  unsigned int num_layers = 0;
  unsigned int batch_size = 0;
  unsigned int max_seq_len = 0;
  unsigned int num_heads_kv = 0;
  unsigned int head_dim = 0;
  ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::FP16;
  ml::train::TensorDim::Format format = ml::train::TensorDim::Format::NCHW;
  KVCacheConfig config;
};

/**
 * @brief Backend interface for physical KV cache storage.
 */
class KVCacheOptimizer {
public:
  virtual ~KVCacheOptimizer() = default;

  virtual void allocate(const KVCacheSpec &spec) = 0;
  virtual bool isAllocated() const = 0;

  virtual nntrainer::Tensor &getKeyCache(unsigned int layer_idx) = 0;
  virtual nntrainer::Tensor &getValueCache(unsigned int layer_idx) = 0;

  virtual nntrainer::Tensor getKeyCacheWriteView(unsigned int layer_idx,
                                                 unsigned int batch,
                                                 unsigned int cache_pos,
                                                 unsigned int step_size) = 0;
  virtual nntrainer::Tensor getValueCacheWriteView(unsigned int layer_idx,
                                                   unsigned int batch,
                                                   unsigned int cache_pos,
                                                   unsigned int step_size) = 0;
  virtual nntrainer::Tensor getKeyCacheReadView(unsigned int layer_idx,
                                                unsigned int batch,
                                                unsigned int read_len) = 0;
  virtual nntrainer::Tensor getValueCacheReadView(unsigned int layer_idx,
                                                  unsigned int batch,
                                                  unsigned int read_len) = 0;

  virtual void save(std::ostream &out, unsigned int seq_len) const = 0;
  virtual void load(std::ifstream &in, unsigned int seq_len) = 0;
};

/**
 * @brief Raw tensor-backed KV cache backend preserving existing behavior.
 */
class RawKVCacheOptimizer final : public KVCacheOptimizer {
public:
  RawKVCacheOptimizer() = default;
  ~RawKVCacheOptimizer() override = default;

  void allocate(const KVCacheSpec &spec) override;
  bool isAllocated() const override { return !layer_caches_.empty(); }

  nntrainer::Tensor &getKeyCache(unsigned int layer_idx) override;
  nntrainer::Tensor &getValueCache(unsigned int layer_idx) override;

  nntrainer::Tensor getKeyCacheWriteView(unsigned int layer_idx,
                                         unsigned int batch,
                                         unsigned int cache_pos,
                                         unsigned int step_size) override;
  nntrainer::Tensor getValueCacheWriteView(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int cache_pos,
                                           unsigned int step_size) override;
  nntrainer::Tensor getKeyCacheReadView(unsigned int layer_idx,
                                        unsigned int batch,
                                        unsigned int read_len) override;
  nntrainer::Tensor getValueCacheReadView(unsigned int layer_idx,
                                          unsigned int batch,
                                          unsigned int read_len) override;

  void save(std::ostream &out, unsigned int seq_len) const override;
  void load(std::ifstream &in, unsigned int seq_len) override;

private:
  struct LayerCache {
    nntrainer::Tensor key_cache;   /**< (batch, 1, max_seq_len, kv_width) */
    nntrainer::Tensor value_cache; /**< (batch, 1, max_seq_len, kv_width) */
  };

  std::vector<LayerCache> layer_caches_;

  unsigned int batch_size_ = 0;
  unsigned int max_seq_len_ = 0;
  unsigned int num_heads_kv_ = 0;
  unsigned int head_dim_ = 0;
  unsigned int kv_width_ = 0;

  ml::train::TensorDim::DataType dtype_ = ml::train::TensorDim::DataType::FP16;
  ml::train::TensorDim::Format format_ = ml::train::TensorDim::Format::NCHW;
};

} // namespace causallm

#endif // __KV_CACHE_OPTIMIZER_H__
