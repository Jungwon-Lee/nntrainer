/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_router_prefetch_layer.cpp
 * @date   30 July 2026
 * @brief  SmallThinker pre-attention router and expert prefetch support.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <algorithm>
#include <cmath>
#include <node_exporter.h>
#include <smallthinker_router_prefetch_layer.h>
#include <stdexcept>
#include <unordered_set>

namespace causallm {

namespace {

void normalizeRouterWeights(nntrainer::Tensor &weights,
                            unsigned int total_tokens, unsigned int topk,
                            bool apply_softmax) {
  float *data = weights.getData<float>();
  for (unsigned int token = 0; token < total_tokens; ++token) {
    float *row = data + token * topk;
    float denominator = 0.0f;

    if (apply_softmax) {
      const float maximum = *std::max_element(row, row + topk);
      for (unsigned int k = 0; k < topk; ++k) {
        row[k] = std::exp(row[k] - maximum);
        denominator += row[k];
      }
    } else {
      for (unsigned int k = 0; k < topk; ++k) {
        row[k] = 1.0f / (1.0f + std::exp(-row[k]));
        denominator += row[k];
      }
    }

    if (denominator != 0.0f) {
      for (unsigned int k = 0; k < topk; ++k)
        row[k] /= denominator;
    }
  }
}

std::mutex state_registry_mutex;
std::unordered_map<std::string, std::weak_ptr<SmallThinkerExpertPrefetchState>>
  state_registry;

} // namespace

SmallThinkerExpertPrefetchState::~SmallThinkerExpertPrefetchState() {
  shutdown();
}

void SmallThinkerExpertPrefetchState::registerWeights(
  nntrainer::Tensor *router,
  std::vector<SmallThinkerExpertWeights> expert_weights,
  unsigned int cache_size) {
  std::lock_guard<std::mutex> lock(mutex);
  NNTR_THROW_IF(shutting_down, std::logic_error)
    << "Cannot register SmallThinker expert weights during shutdown";
  if (router_weight != nullptr)
    return;

  router_weight = router;
  weights = std::move(expert_weights);
  status.assign(weights.size(), Status::UNLOADED);
  pin_count.assign(weights.size(), 0);
  errors.resize(weights.size());
  capacity = cache_size;
}

nntrainer::Tensor *SmallThinkerExpertPrefetchState::getRouterWeight() {
  std::lock_guard<std::mutex> lock(mutex);
  return router_weight;
}

void SmallThinkerExpertPrefetchState::touch(unsigned int expert) {
  auto found = lru_position.find(expert);
  if (found != lru_position.end())
    lru.erase(found->second);
  lru.push_back(expert);
  lru_position[expert] = --lru.end();
}

void SmallThinkerExpertPrefetchState::activate(unsigned int expert) {
  bool gate_activation_attempted = false;
  bool up_activation_attempted = false;
  bool down_activation_attempted = false;
  try {
    gate_activation_attempted = true;
    weights[expert].gate->activate();
    up_activation_attempted = true;
    weights[expert].up->activate();
    down_activation_attempted = true;
    weights[expert].down->activate();

    {
      std::lock_guard<std::mutex> lock(mutex);
      status[expert] = Status::RESIDENT;
      touch(expert);
    }
  } catch (...) {
    const std::exception_ptr activation_error = std::current_exception();
    try {
      if (down_activation_attempted)
        weights[expert].down->deactivate();
    } catch (...) {
    }
    try {
      if (up_activation_attempted)
        weights[expert].up->deactivate();
    } catch (...) {
    }
    try {
      if (gate_activation_attempted)
        weights[expert].gate->deactivate();
    } catch (...) {
    }

    std::lock_guard<std::mutex> lock(mutex);
    errors[expert] = activation_error;
    status[expert] = Status::FAILED;
  }
  condition.notify_all();
}

void SmallThinkerExpertPrefetchState::prefetch(
  const std::vector<unsigned int> &experts) {
  std::lock_guard<std::mutex> task_lock(task_mutex);
  if (prefetch_task.valid())
    prefetch_task.wait();

  std::vector<unsigned int> to_load;
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (router_weight == nullptr || shutting_down)
      return;

    std::unordered_set<unsigned int> unique;
    for (unsigned int expert : experts) {
      if (expert >= weights.size() || !unique.insert(expert).second)
        continue;
      condition.wait(lock, [&]() {
        return status[expert] != Status::EVICTING || shutting_down;
      });
      if (shutting_down)
        return;
      if (status[expert] == Status::UNLOADED) {
        status[expert] = Status::LOADING;
        to_load.push_back(expert);
      } else if (status[expert] == Status::RESIDENT) {
        touch(expert);
      }
    }
  }

  if (to_load.empty())
    return;

  try {
    prefetch_task = std::async(std::launch::async, [this, to_load]() {
      for (unsigned int expert : to_load)
        activate(expert);
    });
  } catch (...) {
    const std::exception_ptr launch_error = std::current_exception();
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (unsigned int expert : to_load) {
        if (status[expert] != Status::LOADING)
          continue;
        errors[expert] = launch_error;
        status[expert] = Status::FAILED;
      }
    }
    condition.notify_all();
    throw;
  }
}

bool SmallThinkerExpertPrefetchState::acquire(unsigned int expert) {
  bool cache_miss = false;
  std::unique_lock<std::mutex> lock(mutex);
  NNTR_THROW_IF(expert >= weights.size(), std::out_of_range)
    << "SmallThinker expert index is out of range";
  NNTR_THROW_IF(shutting_down, std::runtime_error)
    << "SmallThinker expert cache is shutting down";

  while (status[expert] == Status::LOADING ||
         status[expert] == Status::EVICTING) {
    cache_miss = true;
    condition.wait(lock, [&]() {
      return (status[expert] != Status::LOADING &&
              status[expert] != Status::EVICTING) ||
             shutting_down;
    });
    NNTR_THROW_IF(shutting_down, std::runtime_error)
      << "SmallThinker expert cache is shutting down";
  }

  if (status[expert] == Status::UNLOADED) {
    status[expert] = Status::LOADING;
    cache_miss = true;
    lock.unlock();
    activate(expert);
    lock.lock();
  }

  NNTR_THROW_IF(shutting_down, std::runtime_error)
    << "SmallThinker expert cache is shutting down";
  if (status[expert] == Status::FAILED)
    std::rethrow_exception(errors[expert]);

  ++pin_count[expert];
  touch(expert);
  return cache_miss;
}

void SmallThinkerExpertPrefetchState::release(unsigned int expert) {
  std::lock_guard<std::mutex> lock(mutex);
  NNTR_THROW_IF(expert >= pin_count.size() || pin_count[expert] == 0,
                std::logic_error)
    << "SmallThinker expert cache pin count underflow";
  --pin_count[expert];
  condition.notify_all();
}

void SmallThinkerExpertPrefetchState::trim() {
  while (true) {
    SmallThinkerExpertWeights target{};
    unsigned int target_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (shutting_down || lru.size() <= capacity)
        return;

      auto candidate =
        std::find_if(lru.begin(), lru.end(), [&](unsigned int i) {
          return status[i] == Status::RESIDENT && pin_count[i] == 0;
        });
      if (candidate == lru.end())
        return;

      target_index = *candidate;
      target = weights[target_index];
      lru_position.erase(target_index);
      lru.erase(candidate);
      status[target_index] = Status::EVICTING;
    }

    std::exception_ptr eviction_error;
    try {
      target.gate->deactivate();
    } catch (...) {
      eviction_error = std::current_exception();
    }
    try {
      target.up->deactivate();
    } catch (...) {
      if (eviction_error == nullptr)
        eviction_error = std::current_exception();
    }
    try {
      target.down->deactivate();
    } catch (...) {
      if (eviction_error == nullptr)
        eviction_error = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      if (eviction_error == nullptr) {
        status[target_index] = Status::UNLOADED;
      } else {
        errors[target_index] = eviction_error;
        status[target_index] = Status::FAILED;
      }
    }
    condition.notify_all();
  }
}

void SmallThinkerExpertPrefetchState::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    shutting_down = true;
  }
  condition.notify_all();

  std::lock_guard<std::mutex> task_lock(task_mutex);
  if (prefetch_task.valid())
    prefetch_task.wait();

  std::unique_lock<std::mutex> lock(mutex);
  condition.wait(lock, [&]() {
    const bool no_weight_operation =
      std::none_of(status.begin(), status.end(), [](Status current) {
        return current == Status::LOADING || current == Status::EVICTING;
      });
    const bool no_pinned_expert =
      std::none_of(pin_count.begin(), pin_count.end(),
                   [](unsigned int count) { return count != 0; });
    return no_weight_operation && no_pinned_expert;
  });
}

size_t SmallThinkerExpertPrefetchState::residentCount() const {
  std::lock_guard<std::mutex> lock(mutex);
  return lru.size();
}

std::shared_ptr<SmallThinkerExpertPrefetchState>
getSmallThinkerExpertPrefetchState(const std::string &key) {
  NNTR_THROW_IF(key.empty(), std::invalid_argument)
    << "SmallThinker expert prefetch key cannot be empty";

  std::lock_guard<std::mutex> lock(state_registry_mutex);
  auto state = state_registry[key].lock();
  if (state == nullptr) {
    state = std::make_shared<SmallThinkerExpertPrefetchState>();
    state_registry[key] = state;
  }
  return state;
}

void shutdownSmallThinkerExpertPrefetchState(const std::string &key) {
  std::shared_ptr<SmallThinkerExpertPrefetchState> state;
  {
    std::lock_guard<std::mutex> lock(state_registry_mutex);
    auto found = state_registry.find(key);
    if (found == state_registry.end())
      return;
    state = found->second.lock();
  }

  if (state != nullptr)
    state->shutdown();
}

SmallThinkerRouterPrefetchLayer::SmallThinkerRouterPrefetchLayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  router_apply_softmax(true),
  router_props(props::NumExperts(), props::NumExpertsPerToken(),
               props::MoERouterApplySoftmax(), props::MoEPrefetchKey()) {}

void SmallThinkerRouterPrefetchLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "SmallThinker router prefetch layer requires one input";

  num_experts = std::get<props::NumExperts>(router_props).get();
  topk = std::get<props::NumExpertsPerToken>(router_props).get();
  router_apply_softmax =
    std::get<props::MoERouterApplySoftmax>(router_props).get();
  prefetch_state = getSmallThinkerExpertPrefetchState(
    std::get<props::MoEPrefetchKey>(router_props).get());

  const auto &input_dim = context.getInputDimensions()[0];
  auto route_dim = input_dim;
  route_dim.width(topk);
  route_dim.setTensorType(
    {context.getFormat(), nntrainer::TensorDim::DataType::FP32});
  auto index_dim = route_dim;
  index_dim.setTensorType(
    {context.getFormat(), nntrainer::TensorDim::DataType::UINT32});
  auto valid_dim = route_dim;
  valid_dim.width(1);

  context.setOutputDimensions({input_dim, route_dim, index_dim, valid_dim});
}

void SmallThinkerRouterPrefetchLayer::route(const nntrainer::Tensor &input,
                                            nntrainer::Tensor &passthrough,
                                            nntrainer::Tensor &routing_weights,
                                            nntrainer::Tensor &routing_indices,
                                            nntrainer::Tensor &valid) {
  passthrough.copyData(input);
  routing_weights.setZero();
  routing_indices.setZero();
  valid.setZero();

  nntrainer::Tensor *router_weight = prefetch_state->getRouterWeight();
  if (router_weight == nullptr)
    return;

  const unsigned int total_tokens = input.batch() * input.height();
  const unsigned int hidden_size = input.width();
  nntrainer::TensorDim flat_dim({total_tokens, 1, 1, hidden_size},
                                input.getTensorType());
  nntrainer::Tensor flat_input = input.getSharedDataTensor(flat_dim, 0, true);
  nntrainer::TensorDim router_dim(
    {total_tokens, 1, 1, num_experts},
    nntrainer::TensorDim::TensorType(input.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  nntrainer::Tensor router_logits(router_dim);
  flat_input.dot(*router_weight, router_logits);

  auto topk_result = router_logits.topK(topk);
  nntrainer::Tensor topk_values = std::get<0>(topk_result);
  nntrainer::Tensor topk_indices = std::get<1>(topk_result);
  normalizeRouterWeights(topk_values, total_tokens, topk, router_apply_softmax);

  routing_weights.copyData(topk_values);
  routing_indices.copyData(topk_indices);
  std::fill(valid.getData<float>(),
            valid.getData<float>() + valid.getDim().getDataLen(), 1.0f);

  const uint32_t *indices = topk_indices.getData<uint32_t>();
  std::vector<unsigned int> selected(indices, indices + total_tokens * topk);
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  prefetch_state->prefetch(selected);
}

void SmallThinkerRouterPrefetchLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  route(context.getInput(0), context.getOutput(0), context.getOutput(1),
        context.getOutput(2), context.getOutput(3));
}

void SmallThinkerRouterPrefetchLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const nntrainer::Tensor &input = context.getInput(0);
  const unsigned int step = to - from;
  const unsigned int hidden_size = input.width();

  nntrainer::TensorDim input_step_dim({1, 1, step, hidden_size},
                                      input.getTensorType());
  nntrainer::TensorDim route_step_dim({1, 1, step, topk},
                                      context.getOutput(1).getTensorType());
  nntrainer::TensorDim index_step_dim({1, 1, step, topk},
                                      context.getOutput(2).getTensorType());
  nntrainer::TensorDim valid_step_dim({1, 1, step, 1},
                                      context.getOutput(3).getTensorType());

  for (unsigned int batch = 0; batch < input.batch(); ++batch) {
    auto input_step = input.getSharedDataTensor(
      input_step_dim, batch * input_step_dim.getFeatureLen(), true);
    auto passthrough_step = context.getOutput(0).getSharedDataTensor(
      input_step_dim, batch * input_step_dim.getFeatureLen(), true);
    auto weight_step = context.getOutput(1).getSharedDataTensor(
      route_step_dim, batch * route_step_dim.getFeatureLen(), true);
    auto index_step = context.getOutput(2).getSharedDataTensor(
      index_step_dim, batch * index_step_dim.getFeatureLen(), true);
    auto valid_step = context.getOutput(3).getSharedDataTensor(
      valid_step_dim, batch * valid_step_dim.getFeatureLen(), true);
    route(input_step, passthrough_step, weight_step, index_step, valid_step);
  }
}

void SmallThinkerRouterPrefetchLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, router_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void SmallThinkerRouterPrefetchLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "SmallThinker router prefetch does not support derivatives");
}

void SmallThinkerRouterPrefetchLayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "SmallThinker router prefetch does not support gradients");
}

void SmallThinkerRouterPrefetchLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(router_props, method, this);
}

void SmallThinkerRouterPrefetchLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  const nntrainer::TensorDim &input_dim = input_dimensions[0];
  context.updateInput(0, input_dim);

  nntrainer::TensorDim passthrough_dim = input_dim;
  nntrainer::TensorDim route_dim = context.getOutput(1).getDim();
  nntrainer::TensorDim index_dim = context.getOutput(2).getDim();
  nntrainer::TensorDim valid_dim = context.getOutput(3).getDim();
  route_dim.batch(input_dim.batch());
  route_dim.channel(input_dim.channel());
  route_dim.height(input_dim.height());
  index_dim.batch(input_dim.batch());
  index_dim.channel(input_dim.channel());
  index_dim.height(input_dim.height());
  valid_dim.batch(input_dim.batch());
  valid_dim.channel(input_dim.channel());
  valid_dim.height(input_dim.height());
  context.updateOutput(0, passthrough_dim);
  context.updateOutput(1, route_dim);
  context.updateOutput(2, index_dim);
  context.updateOutput(3, valid_dim);
}

} // namespace causallm
