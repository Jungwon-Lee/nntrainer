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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace causallm {

namespace {

constexpr float int8_quant_max = 127.0f;

size_t getScaleIndex(const Int8KVCacheBlock &block, unsigned int token,
                     unsigned int head) {
  if (block.scale_granularity == "per_token_per_head") {
    return static_cast<size_t>(token) * block.num_heads_kv + head;
  }

  if (block.scale_granularity == "per_token") {
    return token;
  }

  if (block.scale_granularity == "per_tensor") {
    return 0;
  }

  throw std::invalid_argument(
    "Int8KVCacheQuantizer: unsupported scale_granularity: " +
    block.scale_granularity);
}

size_t getScaleCount(unsigned int tokens, unsigned int num_heads_kv,
                     const std::string &scale_granularity) {
  if (scale_granularity == "per_token_per_head") {
    return static_cast<size_t>(tokens) * num_heads_kv;
  }

  if (scale_granularity == "per_token") {
    return tokens;
  }

  if (scale_granularity == "per_tensor") {
    return 1;
  }

  throw std::invalid_argument(
    "Int8KVCacheQuantizer: unsupported scale_granularity: " +
    scale_granularity);
}

ml::train::TensorDim::DataType
parseMaterializeDType(const std::string &materialize_dtype) {
  if (materialize_dtype.empty() || materialize_dtype == "auto") {
    return ml::train::TensorDim::DataType::NONE;
  }

  if (materialize_dtype == "fp32" || materialize_dtype == "FP32") {
    return ml::train::TensorDim::DataType::FP32;
  }

#ifdef ENABLE_FP16
  if (materialize_dtype == "fp16" || materialize_dtype == "FP16") {
    return ml::train::TensorDim::DataType::FP16;
  }
#endif

  throw std::invalid_argument(
    "Int8KVCacheOptimizer::allocate: unsupported materialize_dtype: " +
    materialize_dtype);
}

template <typename OutType, typename ScaleGetter, typename SourceIndexGetter>
void dequantizeInt8KVCacheTo(const std::vector<int8_t> &src,
                             unsigned int tokens, unsigned int num_heads_kv,
                             unsigned int head_dim, ScaleGetter getScale,
                             SourceIndexGetter getSourceIndex, OutType *out) {
  const unsigned int width = num_heads_kv * head_dim;

  for (unsigned int t = 0; t < tokens; ++t) {
    const size_t out_token = static_cast<size_t>(t) * width;
    for (unsigned int h = 0; h < num_heads_kv; ++h) {
      const float scale = getScale(t, h);
      const size_t head_begin = static_cast<size_t>(h) * head_dim;
      for (unsigned int i = 0; i < head_dim; ++i) {
        const size_t element_offset = head_begin + i;
        out[out_token + element_offset] =
          static_cast<OutType>(
            static_cast<float>(src[getSourceIndex(t, element_offset)]) *
            scale);
      }
    }
  }
}

} // namespace

Int8KVCacheBlock
Int8KVCacheQuantizer::quantize(const float *data, unsigned int tokens,
                               unsigned int num_heads_kv,
                               unsigned int head_dim,
                               const std::string &scale_granularity) {
  if (data == nullptr) {
    throw std::invalid_argument("Int8KVCacheQuantizer::quantize: data is null");
  }
  if (tokens == 0 || num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "Int8KVCacheQuantizer::quantize: dimensions must be > 0");
  }

  Int8KVCacheBlock block;
  block.tokens = tokens;
  block.num_heads_kv = num_heads_kv;
  block.head_dim = head_dim;
  block.scale_granularity = scale_granularity;

  const unsigned int width = block.width();
  block.data.resize(static_cast<size_t>(tokens) * width);
  block.scales.assign(getScaleCount(tokens, num_heads_kv, scale_granularity),
                      1.0f);

  for (unsigned int t = 0; t < tokens; ++t) {
    const unsigned int heads =
      scale_granularity == "per_token_per_head" ? num_heads_kv : 1;
    for (unsigned int h = 0; h < heads; ++h) {
      const size_t begin =
        scale_granularity == "per_token_per_head"
          ? (static_cast<size_t>(t) * width + static_cast<size_t>(h) * head_dim)
          : (static_cast<size_t>(t) * width);
      const size_t len =
        scale_granularity == "per_token_per_head" ? head_dim : width;

      float max_abs = 0.0f;
      for (size_t i = 0; i < len; ++i) {
        max_abs = std::max(max_abs, std::fabs(data[begin + i]));
      }

      const float scale = max_abs > 0.0f ? max_abs / int8_quant_max : 1.0f;
      const size_t scale_idx = getScaleIndex(block, t, h);
      block.scales[scale_idx] = scale;

      for (size_t i = 0; i < len; ++i) {
        const float quantized = std::round(data[begin + i] / scale);
        const float clamped =
          std::max(-int8_quant_max, std::min(int8_quant_max, quantized));
        block.data[begin + i] = static_cast<int8_t>(clamped);
      }
    }
  }

  if (scale_granularity == "per_tensor") {
    float max_abs = 0.0f;
    for (size_t i = 0; i < block.data.size(); ++i) {
      max_abs = std::max(max_abs, std::fabs(data[i]));
    }

    const float scale = max_abs > 0.0f ? max_abs / int8_quant_max : 1.0f;
    block.scales[0] = scale;
    for (size_t i = 0; i < block.data.size(); ++i) {
      const float quantized = std::round(data[i] / scale);
      const float clamped =
        std::max(-int8_quant_max, std::min(int8_quant_max, quantized));
      block.data[i] = static_cast<int8_t>(clamped);
    }
  }

  return block;
}

void Int8KVCacheQuantizer::dequantize(const Int8KVCacheBlock &block,
                                      float *out) {
  if (out == nullptr) {
    throw std::invalid_argument(
      "Int8KVCacheQuantizer::dequantize: output is null");
  }
  if (block.tokens == 0 || block.num_heads_kv == 0 || block.head_dim == 0) {
    throw std::invalid_argument(
      "Int8KVCacheQuantizer::dequantize: invalid block dimensions");
  }

  const unsigned int width = block.width();
  const size_t expected_size = static_cast<size_t>(block.tokens) * width;
  const size_t expected_scales =
    getScaleCount(block.tokens, block.num_heads_kv, block.scale_granularity);
  if (block.data.size() != expected_size ||
      block.scales.size() != expected_scales) {
    throw std::invalid_argument(
      "Int8KVCacheQuantizer::dequantize: invalid block payload size");
  }

  auto get_scale = [&](unsigned int token, unsigned int head) {
    return block.scales[getScaleIndex(block, token, head)];
  };
  auto get_source_index = [width](unsigned int token, size_t element_offset) {
    return static_cast<size_t>(token) * width + element_offset;
  };

  dequantizeInt8KVCacheTo(block.data, block.tokens, block.num_heads_kv,
                          block.head_dim, get_scale, get_source_index, out);
}

void KVCacheOptimizer::appendLayerCache(unsigned int, unsigned int,
                                        unsigned int, nntrainer::Tensor &,
                                        nntrainer::Tensor &) {
  throw std::runtime_error(
    "KVCacheOptimizer::appendLayerCache: runtime cache is not enabled");
}

nntrainer::Tensor KVCacheOptimizer::materializeKeyCache(
  unsigned int, unsigned int, unsigned int, ml::train::TensorDim::DataType,
  ml::train::TensorDim::Format) {
  throw std::runtime_error(
    "KVCacheOptimizer::materializeKeyCache: runtime cache is not enabled");
}

nntrainer::Tensor KVCacheOptimizer::materializeValueCache(
  unsigned int, unsigned int, unsigned int, ml::train::TensorDim::DataType,
  ml::train::TensorDim::Format) {
  throw std::runtime_error(
    "KVCacheOptimizer::materializeValueCache: runtime cache is not enabled");
}

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

void Int8KVCacheOptimizer::allocate(const KVCacheSpec &spec) {
  if (spec.config.scale_granularity != "per_token_per_head" &&
      spec.config.scale_granularity != "per_token") {
    throw std::invalid_argument(
      "Int8KVCacheOptimizer::allocate: unsupported scale_granularity: " +
      spec.config.scale_granularity);
  }

  batch_size_ = spec.batch_size;
  max_seq_len_ = spec.max_seq_len;
  num_heads_kv_ = spec.num_heads_kv;
  head_dim_ = spec.head_dim;
  kv_width_ = spec.num_heads_kv * spec.head_dim;
  scale_granularity_ = spec.config.scale_granularity;
  materialize_dtype_ = parseMaterializeDType(spec.config.materialize_dtype);
  bindable_dtype_ = spec.dtype;
  format_ = spec.format;

  ml::train::TensorDim bindable_dim({batch_size_, 1, 1, kv_width_},
                                    {format_, bindable_dtype_});

  const size_t data_size =
    static_cast<size_t>(batch_size_) * max_seq_len_ * kv_width_;
  const size_t scale_size = getScaleCount();

  layer_caches_.resize(spec.num_layers);
  for (auto &cache : layer_caches_) {
    cache.bindable_key_cache = nntrainer::Tensor(bindable_dim, true);
    cache.bindable_value_cache = nntrainer::Tensor(bindable_dim, true);
    cache.bindable_key_cache.setZero();
    cache.bindable_value_cache.setZero();
    cache.key_cache.assign(data_size, 0);
    cache.value_cache.assign(data_size, 0);
    cache.key_scales.assign(scale_size, 1.0f);
    cache.value_scales.assign(scale_size, 1.0f);
  }
}

nntrainer::Tensor &Int8KVCacheOptimizer::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].bindable_key_cache;
}

nntrainer::Tensor &Int8KVCacheOptimizer::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].bindable_value_cache;
}

nntrainer::Tensor
Int8KVCacheOptimizer::getKeyCacheWriteView(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int cache_pos,
                                           unsigned int step_size) {
  auto &cache = getKeyCache(layer_idx);
  if (batch >= batch_size_ || cache_pos + step_size > cache.height()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getKeyCacheWriteView: runtime cache does not "
      "expose raw logical views");
  }
  return cache.getSharedDataTensor(
    ml::train::TensorDim({1, 1, step_size, kv_width_},
                         {format_, bindable_dtype_}),
    batch * cache.getDim().getFeatureLen() + cache_pos * kv_width_,
    true);
}

nntrainer::Tensor
Int8KVCacheOptimizer::getValueCacheWriteView(unsigned int layer_idx,
                                             unsigned int batch,
                                             unsigned int cache_pos,
                                             unsigned int step_size) {
  auto &cache = getValueCache(layer_idx);
  if (batch >= batch_size_ || cache_pos + step_size > cache.height()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getValueCacheWriteView: runtime cache does not "
      "expose raw logical views");
  }
  return cache.getSharedDataTensor(
    ml::train::TensorDim({1, 1, step_size, kv_width_},
                         {format_, bindable_dtype_}),
    batch * cache.getDim().getFeatureLen() + cache_pos * kv_width_,
    true);
}

nntrainer::Tensor
Int8KVCacheOptimizer::getKeyCacheReadView(unsigned int layer_idx,
                                          unsigned int batch,
                                          unsigned int read_len) {
  auto &cache = getKeyCache(layer_idx);
  if (batch >= batch_size_ || read_len > cache.height()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getKeyCacheReadView: runtime cache does not "
      "expose raw logical views");
  }
  return cache.getSharedDataTensor(
    ml::train::TensorDim({1, 1, read_len, kv_width_},
                         {format_, bindable_dtype_}),
    batch * cache.getDim().getFeatureLen(), true);
}

nntrainer::Tensor
Int8KVCacheOptimizer::getValueCacheReadView(unsigned int layer_idx,
                                            unsigned int batch,
                                            unsigned int read_len) {
  auto &cache = getValueCache(layer_idx);
  if (batch >= batch_size_ || read_len > cache.height()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::getValueCacheReadView: runtime cache does not "
      "expose raw logical views");
  }
  return cache.getSharedDataTensor(
    ml::train::TensorDim({1, 1, read_len, kv_width_},
                         {format_, bindable_dtype_}),
    batch * cache.getDim().getFeatureLen(), true);
}

void Int8KVCacheOptimizer::appendLayerCache(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int cache_pos,
                                           nntrainer::Tensor &key_step,
                                           nntrainer::Tensor &value_step) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::appendLayerCache: invalid layer_idx");
  }
  if (batch >= batch_size_ || cache_pos + key_step.height() > max_seq_len_ ||
      key_step.height() != value_step.height() || key_step.width() != kv_width_ ||
      value_step.width() != kv_width_) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::appendLayerCache: invalid cache range");
  }

  auto &cache = layer_caches_[layer_idx];
  quantizeStep(key_step, batch, cache_pos, cache.key_cache, cache.key_scales);
  quantizeStep(value_step, batch, cache_pos, cache.value_cache,
               cache.value_scales);
}

nntrainer::Tensor Int8KVCacheOptimizer::materializeKeyCache(
  unsigned int layer_idx, unsigned int batch, unsigned int read_len,
  ml::train::TensorDim::DataType dtype, ml::train::TensorDim::Format format) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::materializeKeyCache: invalid layer_idx");
  }
  auto &cache = layer_caches_[layer_idx];
  return materialize(cache.key_cache, cache.key_scales, cache.key_scratch, batch,
                     read_len, dtype, format);
}

nntrainer::Tensor Int8KVCacheOptimizer::materializeValueCache(
  unsigned int layer_idx, unsigned int batch, unsigned int read_len,
  ml::train::TensorDim::DataType dtype, ml::train::TensorDim::Format format) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::materializeValueCache: invalid layer_idx");
  }
  auto &cache = layer_caches_[layer_idx];
  return materialize(cache.value_cache, cache.value_scales, cache.value_scratch,
                     batch, read_len, dtype, format);
}

void Int8KVCacheOptimizer::save(std::ostream &,
                                unsigned int) const {
  throw std::runtime_error(
    "Int8KVCacheOptimizer::save: int8 KV cache serialization is not supported");
}

void Int8KVCacheOptimizer::load(std::ifstream &, unsigned int) {
  throw std::runtime_error(
    "Int8KVCacheOptimizer::load: int8 KV cache serialization is not supported");
}

ml::train::TensorDim::DataType
Int8KVCacheOptimizer::getRuntimeMaterializeDataType(
  ml::train::TensorDim::DataType fallback) const {
  return materialize_dtype_ == ml::train::TensorDim::DataType::NONE
           ? fallback
           : materialize_dtype_;
}

size_t Int8KVCacheOptimizer::getDataOffset(unsigned int batch,
                                           unsigned int pos) const {
  return (static_cast<size_t>(batch) * max_seq_len_ + pos) * kv_width_;
}

size_t Int8KVCacheOptimizer::getScaleOffset(unsigned int batch,
                                            unsigned int pos,
                                            unsigned int head) const {
  if (scale_granularity_ == "per_token_per_head") {
    return (static_cast<size_t>(batch) * max_seq_len_ + pos) * num_heads_kv_ +
           head;
  }

  return static_cast<size_t>(batch) * max_seq_len_ + pos;
}

size_t Int8KVCacheOptimizer::getScaleCount() const {
  if (scale_granularity_ == "per_token_per_head") {
    return static_cast<size_t>(batch_size_) * max_seq_len_ * num_heads_kv_;
  }

  return static_cast<size_t>(batch_size_) * max_seq_len_;
}

void Int8KVCacheOptimizer::quantizeStep(const nntrainer::Tensor &step,
                                        unsigned int batch,
                                        unsigned int cache_pos,
                                        std::vector<int8_t> &dst,
                                        std::vector<float> &scales) {
  const unsigned int step_size = step.height();

  auto quantize = [&](const auto *src) {
    for (unsigned int t = 0; t < step_size; ++t) {
      const size_t src_token = static_cast<size_t>(t) * kv_width_;
      const size_t dst_token = getDataOffset(batch, cache_pos + t);

      const unsigned int groups =
        scale_granularity_ == "per_token_per_head" ? num_heads_kv_ : 1;
      for (unsigned int group = 0; group < groups; ++group) {
        const size_t group_begin =
          scale_granularity_ == "per_token_per_head"
            ? static_cast<size_t>(group) * head_dim_
            : 0;
        const size_t group_len =
          scale_granularity_ == "per_token_per_head" ? head_dim_ : kv_width_;

        float max_abs = 0.0f;
        for (size_t i = 0; i < group_len; ++i) {
          const float value =
            static_cast<float>(src[src_token + group_begin + i]);
          max_abs = std::max(max_abs, std::fabs(value));
        }

        const float scale = max_abs > 0.0f ? max_abs / int8_quant_max : 1.0f;
        scales[getScaleOffset(batch, cache_pos + t, group)] = scale;

        for (size_t i = 0; i < group_len; ++i) {
          const float value =
            static_cast<float>(src[src_token + group_begin + i]);
          const float quantized = std::round(value / scale);
          const float clamped =
            std::max(-int8_quant_max, std::min(int8_quant_max, quantized));
          dst[dst_token + group_begin + i] = static_cast<int8_t>(clamped);
        }
      }
    }
  };

  if (step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    quantize(step.getData<float>());
#ifdef ENABLE_FP16
  } else if (step.getDataType() == ml::train::TensorDim::DataType::FP16) {
    quantize(step.getData<_FP16>());
#endif
  } else {
    throw std::invalid_argument(
      "Int8KVCacheOptimizer::quantizeStep: only FP32/FP16 step tensors are "
      "supported");
  }
}

nntrainer::Tensor
Int8KVCacheOptimizer::materialize(std::vector<int8_t> &src,
                                  std::vector<float> &scales,
                                  nntrainer::Tensor &scratch,
                                  unsigned int batch, unsigned int read_len,
                                  ml::train::TensorDim::DataType dtype,
                                  ml::train::TensorDim::Format format) {
  if (batch >= batch_size_ || read_len > max_seq_len_) {
    throw std::out_of_range(
      "Int8KVCacheOptimizer::materialize: invalid cache range");
  }
  if (materialize_dtype_ != ml::train::TensorDim::DataType::NONE &&
      materialize_dtype_ != dtype) {
    throw std::invalid_argument(
      "Int8KVCacheOptimizer::materialize: materialize_dtype must match MHA "
      "activation dtype");
  }

  ml::train::TensorDim scratch_dim({1, 1, read_len, kv_width_},
                                   {format, dtype});
  scratch = nntrainer::Tensor(scratch_dim, true);

  auto get_scale = [&](unsigned int token, unsigned int head) {
    return scales[getScaleOffset(batch, token, head)];
  };
  auto get_source_index = [&](unsigned int token, size_t element_offset) {
    return getDataOffset(batch, token) + element_offset;
  };

  if (dtype == ml::train::TensorDim::DataType::FP32) {
    dequantizeInt8KVCacheTo(src, read_len, num_heads_kv_, head_dim_, get_scale,
                            get_source_index, scratch.getData<float>());
    return scratch;
  }

#ifdef ENABLE_FP16
  if (dtype == ml::train::TensorDim::DataType::FP16) {
    dequantizeInt8KVCacheTo(src, read_len, num_heads_kv_, head_dim_, get_scale,
                            get_source_index, scratch.getData<_FP16>());
    return scratch;
  }
#endif

  throw std::invalid_argument(
    "Int8KVCacheOptimizer::materialize: unsupported materialize dtype");
}

} // namespace causallm
