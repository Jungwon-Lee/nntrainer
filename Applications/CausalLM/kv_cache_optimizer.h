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

#include <fstream>
#include <cstdint>
#include <istream>
#include <ostream>
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
 * @brief INT8 quantized KV cache payload for one contiguous token range.
 */
struct Int8KVCacheBlock {
  std::vector<int8_t> data;
  std::vector<float> scales;
  unsigned int tokens = 0;
  unsigned int num_heads_kv = 0;
  unsigned int head_dim = 0;
  std::string scale_granularity = "per_token_per_head";

  unsigned int width() const { return num_heads_kv * head_dim; }
};

/**
 * @brief Symmetric INT8 quantizer for KV cache ranges.
 */
class Int8KVCacheQuantizer {
public:
  static Int8KVCacheBlock
  quantize(const float *data, unsigned int tokens, unsigned int num_heads_kv,
           unsigned int head_dim,
           const std::string &scale_granularity = "per_token_per_head");

  static void dequantize(const Int8KVCacheBlock &block, float *out);
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

  virtual bool isRuntimeCacheEnabled() const { return false; }
  virtual ml::train::TensorDim::DataType getRuntimeMaterializeDataType(
    ml::train::TensorDim::DataType fallback) const {
    return fallback;
  }
  virtual void appendLayerCache(unsigned int layer_idx, unsigned int batch,
                                unsigned int cache_pos,
                                nntrainer::Tensor &key_step,
                                nntrainer::Tensor &value_step);
  virtual nntrainer::Tensor materializeKeyCache(
    unsigned int layer_idx, unsigned int batch, unsigned int read_len,
    ml::train::TensorDim::DataType dtype,
    ml::train::TensorDim::Format format);
  virtual nntrainer::Tensor materializeValueCache(
    unsigned int layer_idx, unsigned int batch, unsigned int read_len,
    ml::train::TensorDim::DataType dtype,
    ml::train::TensorDim::Format format);

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

/**
 * @brief INT8 KV cache backend with materialize-to-scratch attention path.
 */
class Int8KVCacheOptimizer final : public KVCacheOptimizer {
public:
  Int8KVCacheOptimizer() = default;
  ~Int8KVCacheOptimizer() override = default;

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

  bool isRuntimeCacheEnabled() const override { return true; }
  ml::train::TensorDim::DataType getRuntimeMaterializeDataType(
    ml::train::TensorDim::DataType fallback) const override;
  void appendLayerCache(unsigned int layer_idx, unsigned int batch,
                        unsigned int cache_pos, nntrainer::Tensor &key_step,
                        nntrainer::Tensor &value_step) override;
  nntrainer::Tensor materializeKeyCache(
    unsigned int layer_idx, unsigned int batch, unsigned int read_len,
    ml::train::TensorDim::DataType dtype,
    ml::train::TensorDim::Format format) override;
  nntrainer::Tensor materializeValueCache(
    unsigned int layer_idx, unsigned int batch, unsigned int read_len,
    ml::train::TensorDim::DataType dtype,
    ml::train::TensorDim::Format format) override;

  void save(std::ostream &out, unsigned int seq_len) const override;
  void load(std::ifstream &in, unsigned int seq_len) override;

private:
  struct LayerCache {
    nntrainer::Tensor bindable_key_cache;
    nntrainer::Tensor bindable_value_cache;
    std::vector<int8_t> key_cache;
    std::vector<int8_t> value_cache;
    std::vector<float> key_scales;
    std::vector<float> value_scales;
    nntrainer::Tensor key_scratch;
    nntrainer::Tensor value_scratch;
  };

  size_t getDataOffset(unsigned int batch, unsigned int pos) const;
  size_t getScaleOffset(unsigned int batch, unsigned int pos,
                        unsigned int head) const;
  size_t getScaleCount() const;
  void quantizeStep(const nntrainer::Tensor &step, unsigned int batch,
                    unsigned int cache_pos, std::vector<int8_t> &dst,
                    std::vector<float> &scales);
  nntrainer::Tensor materialize(std::vector<int8_t> &src,
                                std::vector<float> &scales,
                                nntrainer::Tensor &scratch,
                                unsigned int batch, unsigned int read_len,
                                ml::train::TensorDim::DataType dtype,
                                ml::train::TensorDim::Format format);

  std::vector<LayerCache> layer_caches_;

  unsigned int batch_size_ = 0;
  unsigned int max_seq_len_ = 0;
  unsigned int num_heads_kv_ = 0;
  unsigned int head_dim_ = 0;
  unsigned int kv_width_ = 0;
  std::string scale_granularity_ = "per_token_per_head";

  ml::train::TensorDim::DataType materialize_dtype_ =
    ml::train::TensorDim::DataType::NONE;
  ml::train::TensorDim::DataType bindable_dtype_ =
    ml::train::TensorDim::DataType::FP16;
  ml::train::TensorDim::Format format_ = ml::train::TensorDim::Format::NCHW;
};

} // namespace causallm

#endif // __KV_CACHE_OPTIMIZER_H__
