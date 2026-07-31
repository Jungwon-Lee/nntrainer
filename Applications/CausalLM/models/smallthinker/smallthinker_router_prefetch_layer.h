// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_router_prefetch_layer.h
 * @date   30 July 2026
 * @brief  SmallThinker pre-attention router and expert prefetch support.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_ROUTER_PREFETCH_LAYER_H__
#define __SMALLTHINKER_ROUTER_PREFETCH_LAYER_H__
#ifdef __cplusplus

#include <condition_variable>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <smallthinker_moe_layer.h>
#include <unordered_map>
#include <vector>

namespace causallm {

/**
 * @brief Expert weights shared by the pre-attention router and cached MoE.
 */
struct SmallThinkerExpertWeights {
  nntrainer::Tensor *gate;
  nntrainer::Tensor *up;
  nntrainer::Tensor *down;
};

/**
 * @brief Thread-safe expert cache shared between router and cached MoE layers.
 */
class SmallThinkerExpertPrefetchState {
public:
  SmallThinkerExpertPrefetchState() = default;
  ~SmallThinkerExpertPrefetchState();

  SmallThinkerExpertPrefetchState(const SmallThinkerExpertPrefetchState &) =
    delete;
  SmallThinkerExpertPrefetchState &
  operator=(const SmallThinkerExpertPrefetchState &) = delete;

  void registerWeights(nntrainer::Tensor *router,
                       std::vector<SmallThinkerExpertWeights> expert_weights,
                       unsigned int cache_size);
  nntrainer::Tensor *getRouterWeight();
  void prefetch(const std::vector<unsigned int> &experts);
  bool acquire(unsigned int expert);
  void release(unsigned int expert);
  void trim();
  void shutdown();
  size_t residentCount() const;

private:
  enum class Status { UNLOADED, LOADING, RESIDENT, EVICTING, FAILED };

  void activate(unsigned int expert);
  void touch(unsigned int expert);

  mutable std::mutex mutex;
  std::mutex task_mutex;
  std::condition_variable condition;
  nntrainer::Tensor *router_weight = nullptr;
  std::vector<SmallThinkerExpertWeights> weights;
  std::vector<Status> status;
  std::vector<unsigned int> pin_count;
  std::vector<std::exception_ptr> errors;
  std::list<unsigned int> lru;
  std::unordered_map<unsigned int, std::list<unsigned int>::iterator>
    lru_position;
  unsigned int capacity = 0;
  bool shutting_down = false;
  std::future<void> prefetch_task;
};

/**
 * @brief Return the state shared by layers carrying the same key.
 */
std::shared_ptr<SmallThinkerExpertPrefetchState>
getSmallThinkerExpertPrefetchState(const std::string &key);

/**
 * @brief Stop asynchronous work for an existing shared state.
 */
void shutdownSmallThinkerExpertPrefetchState(const std::string &key);

/**
 * @class SmallThinkerRouterPrefetchLayer
 * @brief Routes before attention and starts asynchronous expert activation.
 *
 * Output 0 passes the hidden state through. Outputs 1 and 2 contain routing
 * weights and expert indices, and output 3 indicates whether those outputs are
 * valid. The first invocation falls back to routing in the MoE layer because
 * its persistent weight tensors have not yet been registered.
 */
class SmallThinkerRouterPrefetchLayer : public nntrainer::LayerImpl {
public:
  SmallThinkerRouterPrefetchLayer();
  ~SmallThinkerRouterPrefetchLayer() = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void setProperty(const std::vector<std::string> &values) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  const std::string getType() const override {
    return SmallThinkerRouterPrefetchLayer::type;
  }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "smallthinker_router_prefetch";

private:
  void route(const nntrainer::Tensor &input, nntrainer::Tensor &passthrough,
             nntrainer::Tensor &routing_weights,
             nntrainer::Tensor &routing_indices, nntrainer::Tensor &valid);

  unsigned int num_experts;
  unsigned int topk;
  bool router_apply_softmax;
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             props::MoERouterApplySoftmax, props::MoEPrefetchKey>
    router_props;
  std::shared_ptr<SmallThinkerExpertPrefetchState> prefetch_state;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_ROUTER_PREFETCH_LAYER_H__ */
