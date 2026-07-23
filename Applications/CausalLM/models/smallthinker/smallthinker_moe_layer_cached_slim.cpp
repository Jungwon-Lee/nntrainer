/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_moe_layer_cached_slim.cpp
 * @date   26 June 2026
 * @brief  SmallThinker MoE layer with on-demand expert loading and LRU cache.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <node_exporter.h>
#include <smallthinker_moe_layer_cached_slim.h>
#include <stdexcept>
#include <task_executor.h>
#include <thread_manager.h>
#include <unordered_map>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

namespace causallm {

// ── Global registry: maps layer_id → compute node so the prefetch layer can
//    call prefetchExperts() on the correct SmallThinkerCachedSlimMoELayer. ──
static std::mutex g_registry_mutex;
static std::unordered_map<int, SmallThinkerCachedSlimMoELayer *> g_prefetch_registry;

// ── Number of background prefetch threads. 0 = disabled (synchronous path). ──
// madvise(WILLNEED) triggers kernel asynchronous readahead on all platforms,
// overlapping expert page-in with attention compute. Override with
// NNTR_MOE_PREFETCH_THREADS=0 to disable.
static const int g_prefetch_threads = []() -> int {
  const char *env = std::getenv("NNTR_MOE_PREFETCH_THREADS");
  return env ? std::stoi(env) : 2;
}();


static nntrainer::TaskExecutor &get_prefetch_executor() {
  static nntrainer::TaskExecutor executor(
    "moe_prefetch",
    static_cast<size_t>(std::max(1, g_prefetch_threads)));
  return executor;
}

// Used by SmallThinkerMoEPrefetchLayer to look up the paired compute node.
SmallThinkerCachedSlimMoELayer *get_cached_slim_layer_for_prefetch(int id) {
  if (id < 0)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  auto it = g_prefetch_registry.find(id);
  return it != g_prefetch_registry.end() ? it->second : nullptr;
}

namespace {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Set NNTR_RELU_SPARSITY_LOG=1 at runtime to enable per-layer sparsity stats.
// Evaluated once at first use (static init inside a function would also work).
static const bool kLogSparsity =
  (std::getenv("NNTR_RELU_SPARSITY_LOG") != nullptr);

// [SCRATCH Phase-3.0] Bit-exact sparse-FFN ceiling probe. Counts, over the
// post-ReLU intermediate vector, how many fixed-size groups are FULLY zero:
//  - grp4: 4 consecutive neurons (ARM q4_0x4 up output-col group skip ceiling)
//  - grp8: 8 consecutive neurons (x86 q4_0x8 up group skip ceiling)
//  - blk32: 32 consecutive neurons (down contraction QK4_0 all-zero block skip)
// Accumulated per layer (serial expert loop), printed + reset per layer.
// Remove before merge.
static long long g_grp4_zero = 0, g_grp4_tot = 0;
static long long g_grp8_zero = 0, g_grp8_tot = 0;
static long long g_blk32_zero = 0, g_blk32_tot = 0;

void normalize_router_weights(nntrainer::Tensor &topk_values,
                              unsigned int total_tokens, unsigned int topk,
                              bool router_apply_softmax) {
  if (router_apply_softmax) {
    topk_values.apply(nntrainer::ActiFunc::softmax<float>, topk_values);
    return;
  }

  float *values_data = topk_values.getData<float>();
  for (unsigned int i = 0; i < total_tokens; ++i) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k) {
      float &value = values_data[i * topk + k];
      value = 1.0f / (1.0f + std::exp(-value));
      sum += value;
    }
    for (unsigned int k = 0; k < topk; ++k) {
      values_data[i * topk + k] /= sum;
    }
  }
}

// Collect the top (topk + extra) expert indices across all tokens, used as the
// LRU prediction set. Indices only; routing weights are computed separately.
std::vector<unsigned int> collect_predicted(nntrainer::Tensor &router_logits,
                                            unsigned int total_tokens,
                                            unsigned int topk,
                                            unsigned int num_experts) {
  const unsigned int extra =
    std::min(topk + 5u, num_experts); // a few near-misses kept warm
  auto extra_topk_result = router_logits.topK(extra);
  auto extra_topk_indices = std::get<1>(extra_topk_result);
  const uint32_t *idx = extra_topk_indices.getData<uint32_t>();

  std::vector<unsigned int> predicted;
  predicted.reserve(total_tokens * extra);
  for (unsigned int i = 0; i < total_tokens; ++i)
    for (unsigned int k = 0; k < extra; ++k)
      predicted.push_back(idx[i * extra + k]);
  return predicted;
}

} // namespace

SmallThinkerCachedSlimMoELayer::SmallThinkerCachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  router_apply_softmax(true),
  cache_size(32),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::MoERouterApplySoftmax(), props::MoECacheSize()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()),
  tensors_populated_(false),
  layer_id_(-1),
  gate_tensor_ptr_(nullptr) {}

void SmallThinkerCachedSlimMoELayer::finalize(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "SmallThinker cached-slim MoE layer requires expert input and router "
       "input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  router_apply_softmax =
    std::get<props::MoERouterApplySoftmax>(moe_props).get();
  cache_size = std::get<props::MoECacheSize>(moe_props).get();
  // Optional runtime override for tuning / A-B measurement. cache_size==0
  // reproduces the old behavior (evict every expert after each token).
  if (const char *env = std::getenv("NNTR_MOE_CACHE_SIZE"))
    cache_size = static_cast<unsigned int>(std::stoul(env));
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, true));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
  }

  // All experts start unloaded; the LRU cache populates them on first use.
  need_load.assign(num_experts, true);

  // Prefetch state — resize to num_experts; tensor ptrs filled on first forward.
  in_flight_.assign(num_experts, false);
  expert_gate_tensors_.assign(num_experts, nullptr);
  expert_up_tensors_.assign(num_experts, nullptr);
  expert_down_tensors_.assign(num_experts, nullptr);
  tensors_populated_ = false;
  gate_tensor_ptr_ = nullptr;

  // Parse layer_id from name "layerN_ffn_down" for the prefetch registry.
  {
    const std::string &nm = context.getName();
    const std::string prefix = "layer";
    size_t p = nm.find(prefix);
    if (p != std::string::npos) {
      size_t q = nm.find('_', p + prefix.size());
      if (q != std::string::npos) {
        try {
          layer_id_ = std::stoi(nm.substr(p + prefix.size(), q - p - prefix.size()));
        } catch (...) {
          layer_id_ = -1;
        }
      }
    }
    if (layer_id_ >= 0 && g_prefetch_threads > 0) {
      std::lock_guard<std::mutex> lock(g_registry_mutex);
      g_prefetch_registry[layer_id_] = this;
    }
  }

  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void SmallThinkerCachedSlimMoELayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  // Populate expert Tensor pointers on the first call so the prefetch node
  // can activate() the same Tensor objects that compute will read.
  if (!tensors_populated_) {
    gate_tensor_ptr_ = &context.getWeight(gate_idx);
    for (unsigned int i = 0; i < num_experts; ++i) {
      expert_gate_tensors_[i] = &context.getWeight(expert_gate_proj_indices[i]);
      expert_up_tensors_[i]   = &context.getWeight(expert_up_proj_indices[i]);
      expert_down_tensors_[i] = &context.getWeight(expert_down_proj_indices[i]);
    }
    tensors_populated_ = true;
  }

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input = context.getInput(1);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  router_input.dot(gate_weights, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  normalize_router_weights(topk_values, total_tokens, topk,
                           router_apply_softmax);

  expert_mask.setZero();
  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, total_tokens * topk, [&](size_t idx) {
    const size_t i = idx / topk;
    const size_t k = idx % topk;
    expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
  });

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  std::vector<unsigned int> predicted =
    collect_predicted(router_logits, total_tokens, topk, num_experts);

  long long activate_ns = 0, compute_ns = 0, sparsity_nz = 0,
            sparsity_total = 0;
  for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    run_active_expert(context, input, output, assignments, expert_idx,
                      hidden_size, activate_ns, compute_ns, sparsity_nz,
                      sparsity_total);
  }
  (void)activate_ns;
  (void)compute_ns;
  (void)sparsity_nz;
  (void)sparsity_total;

  touch_predicted(predicted);
  evict_experts(context);

  output.reshape({batch_size, 1, seq_len, hidden_size});
  input.reshape({batch_size, 1, seq_len, hidden_size});
  router_input.reshape({batch_size, 1, seq_len, hidden_size});
}

void SmallThinkerCachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  long long &sparsity_nz, long long &sparsity_total) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    token_input.dot(gate_proj, gate_out);
    acti_func.run_fn(gate_out, acti_out);
    if (kLogSparsity) {
      sparsity_total += intermediate_size;
      const float *d = acti_out.getData<float>();
      for (unsigned j = 0; j < intermediate_size; ++j)
        if (d[j] != 0.0f)
          ++sparsity_nz;
      // [SCRATCH Phase-3.0] fully-zero group/block census on this token's
      // post-ReLU vector (the up-skip and down-skip masks).
      auto count_zero_groups = [&](unsigned g, long long &zero, long long &tot) {
        for (unsigned base = 0; base + g <= intermediate_size; base += g) {
          bool all_zero = true;
          for (unsigned k = 0; k < g; ++k)
            if (d[base + k] != 0.0f) { all_zero = false; break; }
          if (all_zero)
            ++zero;
          ++tot;
        }
      };
      count_zero_groups(4, g_grp4_zero, g_grp4_tot);
      count_zero_groups(8, g_grp8_zero, g_grp8_tot);
      count_zero_groups(32, g_blk32_zero, g_blk32_tot);
    }
    token_input.dot(up_proj, up_out);
    acti_out.multiply_i(up_out);

    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }

  output.add_i(expert_output);
}

bool SmallThinkerCachedSlimMoELayer::run_active_expert(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
  nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &assignments,
  unsigned int expert_idx, unsigned int hidden_size, long long &activate_ns,
  long long &compute_ns, long long &sparsity_nz, long long &sparsity_total) {

  // If a prefetch task is in flight for this expert, wait for it to complete.
  // This is the fast-path when the prefetch already finished — wait() returns
  // immediately and the expert is resident. If the task failed (exception), it
  // clears in_flight_ and leaves need_load=true so we fall through to the
  // synchronous activate below.
  {
    std::unique_lock<std::mutex> lock(cache_mutex);
    cache_cv_.wait(lock, [&] { return !in_flight_[expert_idx]; });
  }

  bool is_miss;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    is_miss = need_load[expert_idx];
  }

  // On a miss (prediction miss or prefetch exception), activate synchronously.
  // activate()/deactivate() must operate on the persistent context weight so
  // that eviction can later munmap the same mapping. We never deactivate here;
  // eviction is deferred to evict_experts().
  if (is_miss) {
    auto ta0 = high_resolution_clock::now();
    context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
    context.getWeight(expert_up_proj_indices[expert_idx]).activate();
    context.getWeight(expert_down_proj_indices[expert_idx]).activate();
    auto ta1 = high_resolution_clock::now();
    activate_ns += duration_cast<nanoseconds>(ta1 - ta0).count();

    std::lock_guard<std::mutex> lock(cache_mutex);
    loaded_expert_deque.push_back(expert_idx);
    iteration_map[expert_idx] = --loaded_expert_deque.end();
    need_load[expert_idx] = false;
  } else {
    // Prefetch hit: expert was activated by the background prefetch worker but
    // not yet in the LRU (it's in prefetch_staged_). Promote it to LRU now
    // so eviction and touch_predicted track it correctly.
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (prefetch_staged_.erase(static_cast<int>(expert_idx))) {
      loaded_expert_deque.push_back(expert_idx);
      iteration_map[expert_idx] = --loaded_expert_deque.end();
    }
    // If not in prefetch_staged_ either, it's a normal LRU hit — no action.
  }

  auto tc0 = high_resolution_clock::now();
  compute_expert_forward(
    input, output, assignments,
    context.getWeight(expert_gate_proj_indices[expert_idx]),
    context.getWeight(expert_up_proj_indices[expert_idx]),
    context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
    sparsity_nz, sparsity_total);
  auto tc1 = high_resolution_clock::now();
  compute_ns += duration_cast<nanoseconds>(tc1 - tc0).count();

  return is_miss;
}

void SmallThinkerCachedSlimMoELayer::touch_predicted(
  const std::vector<unsigned int> &predicted) {
  // Move already-resident predicted experts (this token's top-k plus a few
  // near-misses) to the MRU end so eviction prefers genuinely cold experts.
  std::lock_guard<std::mutex> lock(cache_mutex);
  for (auto it = predicted.rbegin(); it != predicted.rend(); ++it) {
    auto found = iteration_map.find(static_cast<int>(*it));
    if (found != iteration_map.end()) {
      loaded_expert_deque.erase(found->second);
      loaded_expert_deque.push_back(static_cast<int>(*it));
      iteration_map[static_cast<int>(*it)] = --loaded_expert_deque.end();
    }
  }
}

void SmallThinkerCachedSlimMoELayer::evict_experts(
  nntrainer::RunLayerContext &context) {
  // Evict from the main LRU until within budget.
  while (true) {
    int target_idx;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (loaded_expert_deque.size() <= cache_size)
        break;
      target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
    }
    context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
    context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
    context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
  }

  // Evict stale prefetch-staged experts (activated speculatively but not yet
  // used by compute). Keep up to cache_size staged entries to avoid wasting the
  // I/O work; evict beyond that budget using the stored Tensor pointers.
  while (true) {
    int target_idx;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (prefetch_staged_.size() <= cache_size)
        break;
      target_idx = *prefetch_staged_.begin();
      prefetch_staged_.erase(prefetch_staged_.begin());
      need_load[target_idx] = true;
    }
    // Use stored Tensor ptrs — these are the same objects as context.getWeight().
    expert_gate_tensors_[target_idx]->deactivate();
    expert_up_tensors_[target_idx]->deactivate();
    expert_down_tensors_[target_idx]->deactivate();
  }
}

void SmallThinkerCachedSlimMoELayer::prefetchExperts(
  const std::vector<unsigned int> &predicted) {
  if (!tensors_populated_ || g_prefetch_threads == 0)
    return;

  auto &executor = get_prefetch_executor();

  for (unsigned int e : predicted) {
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      // Skip if already resident or already being loaded
      if (!need_load[e] || in_flight_[e])
        continue;
      in_flight_[e] = true;
    }

    nntrainer::Tensor *gate_t = expert_gate_tensors_[e];
    nntrainer::Tensor *up_t   = expert_up_tensors_[e];
    nntrainer::Tensor *down_t = expert_down_tensors_[e];

    // Submit background activate() task. On success, transitions to RESIDENT
    // and pushes into the LRU. On exception, reverts to UNLOADED so compute
    // falls back to synchronous activate.
    executor.submit([this, e, gate_t, up_t, down_t](void *) {
      try {
        gate_t->activate();
        up_t->activate();
        down_t->activate();
        {
          std::lock_guard<std::mutex> lock(cache_mutex);
          need_load[e] = false;
          in_flight_[e] = false;
          // Do NOT add to loaded_expert_deque — that would pollute the LRU with
          // speculative experts and cause evictions of actually-needed entries.
          // Instead, track in prefetch_staged_; compute promotes to LRU on use.
          prefetch_staged_.insert(static_cast<int>(e));
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        in_flight_[e] = false;
      }
      cache_cv_.notify_all();
    });
  }
}

void SmallThinkerCachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // Populate expert Tensor pointers on the first call (decode entry point).
  if (!tensors_populated_) {
    gate_tensor_ptr_ = &context.getWeight(gate_idx);
    for (unsigned int i = 0; i < num_experts; ++i) {
      expert_gate_tensors_[i] = &context.getWeight(expert_gate_proj_indices[i]);
      expert_up_tensors_[i]   = &context.getWeight(expert_up_proj_indices[i]);
      expert_down_tensors_[i] = &context.getWeight(expert_down_proj_indices[i]);
    }
    tensors_populated_ = true;
  }

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input_ = context.getInput(1);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto router_input = router_input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    input.reshape({total_tokens, 1, 1, hidden_size});
    router_input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
    expert_mask.setZero();

    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    router_input.dot(gate_weights, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    normalize_router_weights(topk_values, total_tokens, topk,
                             router_apply_softmax);

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
      }
    }

    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // LRU prediction set (this token's top-k plus a few near-misses), indices
    // only; routing weights above are unchanged so output is bit-identical.
    std::vector<unsigned int> predicted =
      collect_predicted(router_logits, total_tokens, topk, num_experts);

    int hit_count = 0, miss_count = 0;
    long long activate_ns = 0, compute_ns = 0, sparsity_nz = 0,
              sparsity_total = 0;
    auto t_loop0 = high_resolution_clock::now();

    // Serial outer loop: the Q4_0 expert GEMV parallelizes internally via
    // ThreadManager, and nesting parallel_for deadlocks.
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      bool is_miss = run_active_expert(context, input, output, assignments,
                                       expert_idx, hidden_size, activate_ns,
                                       compute_ns, sparsity_nz, sparsity_total);
      if (kLogSparsity)
        is_miss ? ++miss_count : ++hit_count;
      else
        (void)is_miss;
    }

    // Keep predicted experts warm, then evict the coldest beyond cache_size.
    touch_predicted(predicted);
    evict_experts(context);

    if (kLogSparsity) {
      auto t_loop1 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t_loop1 - t_loop0);
      float zero_pct =
        sparsity_total > 0
          ? 100.0f * (sparsity_total - sparsity_nz) / sparsity_total
          : 0.0f;
      // [SCRATCH Phase-3.0] bit-exact skip ceilings: fraction of fully-zero
      // groups = fraction of up cols / down K-blocks we can skip reading.
      auto pct = [](long long z, long long t) {
        return t > 0 ? 100.0f * (float)z / (float)t : 0.0f;
      };
      float up4 = pct(g_grp4_zero, g_grp4_tot);
      float up8 = pct(g_grp8_zero, g_grp8_tot);
      float dn32 = pct(g_blk32_zero, g_blk32_tot);
      std::cout << context.getName() << " \t| " << dt.count() / 1'000'000
                << " ms"
                << "\t| hit=" << hit_count << " miss=" << miss_count
                << "\t| activate=" << activate_ns / 1'000'000 << "ms"
                << " compute=" << compute_ns / 1'000'000 << "ms"
                << " resident=" << loaded_expert_deque.size()
                << "\t| relu_zero=" << zero_pct << "%"
                << "\t| up_skip4=" << up4 << "% up_skip8=" << up8
                << "% down_skip32=" << dn32 << "%" << std::endl;
      g_grp4_zero = g_grp4_tot = 0;
      g_grp8_zero = g_grp8_tot = 0;
      g_blk32_zero = g_blk32_tot = 0;
    }

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void SmallThinkerCachedSlimMoELayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void SmallThinkerCachedSlimMoELayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void SmallThinkerCachedSlimMoELayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void SmallThinkerCachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm
