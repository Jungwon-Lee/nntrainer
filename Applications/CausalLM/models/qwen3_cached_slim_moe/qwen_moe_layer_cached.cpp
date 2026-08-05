/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	qwen_moe_layer_fsu.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 * @note    MoE layer with on-the-fly expert FSU
 *
 */

#include <acti_func.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <node_exporter.h>
#include <qwen_moe_layer_cached.h>
#include <stdexcept>
#include <thread_manager.h>

#include <chrono>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t MAX_CACHED_EXPERTS = 32;
static constexpr size_t PREFETCH_LOOKAHEAD = 2;

namespace {

struct ExpertWeights {
  nntrainer::Tensor *gate;
  nntrainer::Tensor *up;
  nntrainer::Tensor *down;
};

void activateExpert(const ExpertWeights &weights) {
  bool gate_activated = false;
  bool up_activated = false;

  try {
    weights.gate->activate();
    gate_activated = true;
    weights.up->activate();
    up_activated = true;
    weights.down->activate();
  } catch (...) {
    if (up_activated) {
      try {
        weights.up->deactivate();
      } catch (...) {
      }
    }
    if (gate_activated) {
      try {
        weights.gate->deactivate();
      } catch (...) {
      }
    }
    throw;
  }
}

void prefetchExpert(const ExpertWeights &weights) {
  weights.gate->prefetch();
  weights.up->prefetch();
  weights.down->prefetch();
}

void deactivateExpert(const ExpertWeights &weights) {
  std::exception_ptr error;
  try {
    weights.gate->deactivate();
  } catch (...) {
    error = std::current_exception();
  }
  try {
    weights.up->deactivate();
  } catch (...) {
    if (error == nullptr)
      error = std::current_exception();
  }
  try {
    weights.down->deactivate();
  } catch (...) {
    if (error == nullptr)
      error = std::current_exception();
  }
  if (error != nullptr)
    std::rethrow_exception(error);
}

} // namespace

CachedSlimMoELayer::CachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  loaded_expert_deque({}),
  need_load({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void CachedSlimMoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

  // activation function
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

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initializer expert weights
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
    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false, true));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
    need_load.push_back(true);
  }

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

inline void CachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim out_step_dim({1, 1, 1, hidden_size},
                                    input.getTensorType());
  // Create intermediate tensors for this token
  nntrainer::Tensor gate_out(intermediate_dim);
  nntrainer::Tensor acti_out(intermediate_dim);
  nntrainer::Tensor up_out(intermediate_dim);
  nntrainer::Tensor token_input(token_input_dim);
  const unsigned token_idx = token_assignments[0].first;

  if (num_tokens > 1) {
    /** if prefill, copy data to make a batch */
    {
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
        const unsigned token_idx = token_assignments[i].first;
        // Use tensor's optimized copy operation
        nntrainer::Tensor src_view = input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, token_idx * hidden_size, true);
        nntrainer::Tensor dst_view = token_input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, i * hidden_size, true);
        dst_view.copyData(src_view);
      });
    }
  } else {
    /** if token generation, do not copy but get the shared tensor */
    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);
  }

  // Gate projection using optimized dot operation
  token_input.dot(gate_proj, gate_out);

  // Up projection using optimized dot operation
  token_input.dot(up_proj, up_out);

  if (num_tokens == 1) {
    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
                      gate_out.getData<float>(), up_out.getData<float>());
  } else {
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
      const unsigned offset = acti_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_out.getData<float>() + offset,
                        up_out.getData<float>() + offset);
    });
  }

  acti_out.dot(down_proj, output);

  // Apply routing weights to the compact expert output.
  for (size_t i = 0; i < num_tokens; ++i) {
    nntrainer::Tensor expert_token_output =
      output.getSharedDataTensor(out_step_dim, i * hidden_size, true);
    expert_token_output.multiply_i(token_assignments[i].second);
  }
}

void CachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

#ifdef DEBUG
  auto t1 = high_resolution_clock::now();
#endif

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

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
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    // routing
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);

    // Get additional candidates used to keep likely decode experts resident.
    const unsigned int extra_topk_count = std::min(num_experts, topk + 5);
    auto extra_topk_result = router_logits.topK(extra_topk_count);
    auto extra_topk_indices = std::get<1>(extra_topk_result);
    std::deque<int> extra_top_k = {};
    const uint32_t *extra_indices_data = extra_topk_indices.getData<uint32_t>();

    // Store oldest tokens first and the latest token last in LRU update order.
    for (int i = static_cast<int>(total_tokens) - 1; i >= 0; --i) {
      for (int k = 0; k < static_cast<int>(extra_topk_count); ++k) {
        unsigned expert_idx = extra_indices_data[i * extra_topk_count + k];
        extra_top_k.push_back(expert_idx);
      }
    }

    std::vector<bool> retain_expert(num_experts, false);
    const size_t latest_token_offset =
      static_cast<size_t>(total_tokens - 1) * extra_topk_count;
    for (unsigned int k = 0; k < extra_topk_count; ++k)
      retain_expert[extra_indices_data[latest_token_offset + k]] = true;

    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // norm_topk_prob
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    std::vector<int> target_idx_vector;
    target_idx_vector.reserve(num_experts);

    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      target_idx_vector.push_back(expert_idx);
      expert_outputs[expert_idx] =
        nntrainer::Tensor(static_cast<unsigned int>(assignments.size()), 1, 1,
                          hidden_size, output.getTensorType());
    }

    std::vector<int> compute_order = target_idx_vector;
    if (total_tokens > 1) {
      std::stable_sort(compute_order.begin(), compute_order.end(),
                       [&](int lhs, int rhs) {
                         return expert_assignments[lhs].size() >
                                expert_assignments[rhs].size();
                       });
    }

    std::vector<ExpertWeights> expert_weights;
    expert_weights.reserve(num_experts);
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      expert_weights.push_back(
        {&context.getWeight(expert_gate_proj_indices[expert_idx]),
         &context.getWeight(expert_up_proj_indices[expert_idx]),
         &context.getWeight(expert_down_proj_indices[expert_idx])});
    }

    int hit_count = 0;
    int miss_count = 0;

#ifdef DEBUG
    auto t1_miss = high_resolution_clock::now();
    auto t2_miss = t1_miss;
    auto t1_hit = high_resolution_clock::now();
    auto t2_hit = t1_hit;
#endif

    auto remove_from_cache = [&](int expert_idx) {
      loaded_expert_deque.erase(iteration_map.at(expert_idx));
      iteration_map.erase(expert_idx);
      need_load[expert_idx] = true;
    };

    // Serial expert computation is intentional: dot() parallelizes internally.
    // During prefill, a separate storage worker fills the bounded expert cache
    // in workload order while the current expert performs its batched GEMMs.
    if (total_tokens > 1 && !compute_order.empty()) {
      std::condition_variable cache_condition;
      std::vector<bool> pending_expert(num_experts, false);
      std::vector<bool> prefetch_ready(num_experts, false);
      std::vector<size_t> job_rank(num_experts, compute_order.size());
      for (size_t rank = 0; rank < compute_order.size(); ++rank) {
        pending_expert[compute_order[rank]] = true;
        job_rank[compute_order[rank]] = rank;
      }

      bool cancel_prefetch = false;
      int computing_expert = -1;
      size_t next_compute_rank = 0;
      std::exception_ptr prefetch_error;

      auto prefetch_task = std::async(std::launch::async, [&]() {
        try {
          for (size_t rank = 0; rank < compute_order.size(); ++rank) {
            const int expert_idx = compute_order[rank];

            while (true) {
              int eviction_target = -1;
              {
                std::unique_lock<std::mutex> lock(cache_mutex);
                cache_condition.wait(lock, [&]() {
                  return cancel_prefetch ||
                         rank <= next_compute_rank + PREFETCH_LOOKAHEAD;
                });
                if (cancel_prefetch)
                  return;

                if (!need_load[expert_idx]) {
                  lock.unlock();
                  prefetchExpert(expert_weights[expert_idx]);
                  lock.lock();
                  prefetch_ready[expert_idx] = true;
                  ++hit_count;
                  cache_condition.notify_all();
                  break;
                }

                if (loaded_expert_deque.size() < MAX_CACHED_EXPERTS) {
                  lock.unlock();
                  activateExpert(expert_weights[expert_idx]);
                  try {
                    prefetchExpert(expert_weights[expert_idx]);
                  } catch (...) {
                    deactivateExpert(expert_weights[expert_idx]);
                    throw;
                  }
                  lock.lock();
                  loaded_expert_deque.push_back(expert_idx);
                  iteration_map[expert_idx] = --loaded_expert_deque.end();
                  need_load[expert_idx] = false;
                  prefetch_ready[expert_idx] = true;
                  ++miss_count;
                  cache_condition.notify_all();
                  break;
                }

                auto find_candidate = [&](bool allow_retained,
                                          bool allow_future) {
                  return std::find_if(
                    loaded_expert_deque.begin(), loaded_expert_deque.end(),
                    [&](int candidate) {
                      if (candidate == computing_expert)
                        return false;
                      if (!allow_retained && retain_expert[candidate])
                        return false;
                      if (!pending_expert[candidate])
                        return true;
                      return allow_future && job_rank[candidate] >= rank;
                    });
                };

                auto candidate = find_candidate(false, false);
                if (candidate == loaded_expert_deque.end() &&
                    rank == next_compute_rank)
                  candidate = find_candidate(false, true);
                if (candidate == loaded_expert_deque.end() &&
                    rank == next_compute_rank)
                  candidate = find_candidate(true, false);
                if (candidate == loaded_expert_deque.end() &&
                    rank == next_compute_rank)
                  candidate = find_candidate(true, true);

                if (candidate == loaded_expert_deque.end()) {
                  cache_condition.wait(lock);
                  continue;
                }

                eviction_target = *candidate;
                remove_from_cache(eviction_target);
              }

              deactivateExpert(expert_weights[eviction_target]);
            }
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(cache_mutex);
          prefetch_error = std::current_exception();
          cache_condition.notify_all();
        }
      });

      try {
        for (int expert_idx : compute_order) {
          {
            std::unique_lock<std::mutex> lock(cache_mutex);
            cache_condition.wait(lock, [&]() {
              return prefetch_ready[expert_idx] || prefetch_error != nullptr;
            });
            if (prefetch_error != nullptr)
              std::rethrow_exception(prefetch_error);
            computing_expert = expert_idx;
          }

          compute_expert_forward(
            input, expert_outputs[expert_idx], expert_assignments[expert_idx],
            *expert_weights[expert_idx].gate, *expert_weights[expert_idx].up,
            *expert_weights[expert_idx].down, hidden_size);

          {
            std::lock_guard<std::mutex> lock(cache_mutex);
            computing_expert = -1;
            pending_expert[expert_idx] = false;
            ++next_compute_rank;
          }
          cache_condition.notify_all();
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(cache_mutex);
          cancel_prefetch = true;
          computing_expert = -1;
          std::fill(pending_expert.begin(), pending_expert.end(), false);
        }
        cache_condition.notify_all();
        if (prefetch_task.valid()) {
          try {
            prefetch_task.get();
          } catch (...) {
          }
        }
        throw;
      }

      prefetch_task.get();
    } else {
      for (int expert_idx : compute_order) {
        const auto &assignments = expert_assignments[expert_idx];
        if (need_load[expert_idx]) {
#ifdef DEBUG
          t1_miss = high_resolution_clock::now();
#endif
          int eviction_target = -1;
          {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (loaded_expert_deque.size() >= MAX_CACHED_EXPERTS) {
              auto candidate = std::find_if(
                loaded_expert_deque.begin(), loaded_expert_deque.end(),
                [&](int current) { return !retain_expert[current]; });
              if (candidate == loaded_expert_deque.end())
                candidate = loaded_expert_deque.begin();
              eviction_target = *candidate;
              remove_from_cache(eviction_target);
            }
          }
          if (eviction_target >= 0)
            deactivateExpert(expert_weights[eviction_target]);

          activateExpert(expert_weights[expert_idx]);
          {
            std::lock_guard<std::mutex> lock(cache_mutex);
            loaded_expert_deque.push_back(expert_idx);
            iteration_map[expert_idx] = --loaded_expert_deque.end();
            need_load[expert_idx] = false;
            ++miss_count;
          }
#ifdef DEBUG
          t2_miss = high_resolution_clock::now();
#endif
        } else {
#ifdef DEBUG
          t1_hit = high_resolution_clock::now();
#endif
          ++hit_count;
#ifdef DEBUG
          t2_hit = high_resolution_clock::now();
#endif
        }

        compute_expert_forward(input, expert_outputs[expert_idx], assignments,
                               *expert_weights[expert_idx].gate,
                               *expert_weights[expert_idx].up,
                               *expert_weights[expert_idx].down, hidden_size);
      }
    }

    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      for (auto iter = extra_top_k.rbegin(); iter != extra_top_k.rend();
           ++iter) {
        auto found = iteration_map.find(*iter);
        if (found != iteration_map.end()) {
          loaded_expert_deque.erase(found->second);
          loaded_expert_deque.push_back(*iter);
          iteration_map[*iter] = --loaded_expert_deque.end();
        }
      }
    }

#ifdef DEBUG
    auto t1_evict = high_resolution_clock::now();
#endif

    // Evict experts
    /// @todo apply multi thread loop
    while (loaded_expert_deque.size() > MAX_CACHED_EXPERTS) {
      int target_idx;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        target_idx = loaded_expert_deque.front();
        loaded_expert_deque.pop_front();
        iteration_map.erase(target_idx);
        need_load[target_idx] = true;
      }

      context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
    }

#ifdef DEBUG
    auto t2_evict = high_resolution_clock::now();
#endif

    // Combine expert outputs
    nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                        output.getTensorType());
    for (int expert_idx : target_idx_vector) {
      const auto &assignments = expert_assignments[expert_idx];
      for (size_t i = 0; i < assignments.size(); ++i) {
        nntrainer::Tensor token_output = output.getSharedDataTensor(
          token_step_dim, assignments[i].first * hidden_size, true);
        nntrainer::Tensor expert_token_output =
          expert_outputs[expert_idx].getSharedDataTensor(token_step_dim,
                                                         i * hidden_size, true);
        token_output.add_i(expert_token_output);
      }
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});

#ifdef DEBUG
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    auto dt_miss = duration_cast<nanoseconds>(t2_miss - t1_miss);
    auto dt_hit = duration_cast<nanoseconds>(t2_hit - t1_hit);
    auto dt_evict = duration_cast<nanoseconds>(t2_evict - t1_evict);
    std::cout << context.getName() << " \t| " << dt.count() << " ns "
              << "\t| " << dt.count() / 1'000 << " us "
              << "\t| " << dt.count() / 1'000'000 << " ms "
              << "\t| "
              << "hit ratio: " << hit_count / 8.0 << "\t | "
              << " miss ratio: " << miss_count / 8.0 << "\t | "
              << "hit_compute: " << dt_hit.count() / 1'000'000 << " ms "
              << "\t| "
              << "miss_compute: " << dt_miss.count() / 1'000'000 << " ms "
              << "\t| "
              << "evict_time: " << dt_evict.count() / 1'000'000 << " ms "
              << "\t| " << std::endl;
#endif
  }
}

void CachedSlimMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void CachedSlimMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void CachedSlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void CachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

void CachedSlimMoELayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

} // namespace causallm
