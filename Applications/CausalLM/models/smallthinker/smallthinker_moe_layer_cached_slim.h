// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_cached_slim.h
 * @date   26 June 2026
 * @brief  SmallThinker MoE layer with on-demand expert loading and LRU cache.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__
#ifdef __cplusplus

#include <list>
#include <mutex>
#include <smallthinker_moe_layer.h>
#include <unordered_map>
#include <vector>

namespace causallm {

/**
 * @class   SmallThinkerCachedSlimMoELayer
 * @brief   SmallThinker Mixture of Expert Layer with virtual expert weights
 *          and an LRU cache to keep recently-used experts mapped across tokens.
 */
class SmallThinkerCachedSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of SmallThinker cached-slim Mixture of Expert Layer
   */
  SmallThinkerCachedSlimMoELayer();

  /**
   * @brief     Destructor of SmallThinker cached-slim Mixture of Expert Layer
   */
  ~SmallThinkerCachedSlimMoELayer() = default;

  /**
   * @brief  Move constructor.
   * @param[in] SmallThinkerCachedSlimMoELayer &&
   */
  SmallThinkerCachedSlimMoELayer(
    SmallThinkerCachedSlimMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs SmallThinkerCachedSlimMoELayer to be moved.
   */
  SmallThinkerCachedSlimMoELayer &
  operator=(SmallThinkerCachedSlimMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerCachedSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type =
    "smallthinker_moe_cached_slim"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token */
  bool router_apply_softmax;     /**< whether router uses softmax or sigmoid */
  unsigned int cache_size;       /**< max resident experts in the LRU cache */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             props::MoERouterApplySoftmax, props::MoECacheSize>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  // LRU cache of resident (mmap'd) experts. An expert stays mapped across
  // tokens until evicted, so repeatedly-hit experts are not re-loaded. Guarded
  // by cache_mutex so a background prefetch thread can mutate it too.
  std::list<int> loaded_expert_deque; /**< LRU order */
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::vector<bool> need_load; /**< per-expert: must activate before use */
  std::mutex cache_mutex;

  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief Activate the expert on a cache miss (and record it in the LRU),
   *        then run the expert forward. Does NOT deactivate; eviction is
   *        deferred to evict_experts().
   * @return true if this was a cache miss (expert was activated here).
   */
  bool
  run_active_expert(nntrainer::RunLayerContext &context,
                    const nntrainer::Tensor &input, nntrainer::Tensor &output,
                    const std::vector<std::pair<unsigned, float>> &assignments,
                    unsigned int expert_idx, unsigned int hidden_size,
                    long long &activate_ns, long long &compute_ns);

  /** @brief Bump predicted experts to the MRU end so they survive eviction. */
  void touch_predicted(const std::vector<unsigned int> &predicted);

  /** @brief Deactivate (munmap) LRU-front experts until size <= cache_size. */
  void evict_experts(nntrainer::RunLayerContext &context);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__ */
