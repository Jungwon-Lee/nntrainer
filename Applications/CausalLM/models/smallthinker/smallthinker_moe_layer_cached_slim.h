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

#include <condition_variable>
#include <list>
#include <mutex>
#include <smallthinker_moe_layer.h>
#include <unordered_map>
#include <unordered_set>
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

  // Members below are protected (not private) so the sparse subclass
  // (SmallThinkerSparseCachedSlimMoELayer) can reuse the LRU / prefetch
  // machinery and override compute_expert_forward().

  /**
   * @brief Return the gate (router) weight pointer; used by the prefetch node
   *        to compute router logits at block entry without owning the weight.
   *        Returns nullptr until the first forward pass populates it.
   */
  nntrainer::Tensor *getGateTensor() const {
    return tensors_populated_
             ? const_cast<nntrainer::Tensor *>(gate_tensor_ptr_)
             : nullptr;
  }

  /**
   * @brief Fire background activate() tasks for predicted experts. Called by
   *        the prefetch node before attention runs. No-op until the first
   *        forward pass populates the Tensor pointers (cold start: token 0).
   */
  void prefetchExperts(const std::vector<unsigned int> &predicted);

protected:
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
  std::condition_variable cache_cv_; /**< notified when in_flight_ clears */

  // Expert-preload state (Phase 2). Populated lazily on the first forward pass.
  std::vector<bool> in_flight_;                          /**< background activate in progress */
  std::vector<nntrainer::Tensor *> expert_gate_tensors_; /**< ptrs into compute context */
  std::vector<nntrainer::Tensor *> expert_up_tensors_;
  std::vector<nntrainer::Tensor *> expert_down_tensors_;
  std::unordered_set<int> prefetch_staged_; /**< activated by prefetch, not yet in LRU */
  bool tensors_populated_; /**< true after first forward fills tensor ptrs */
  int layer_id_;           /**< parsed from layer name; registry key */
  const nntrainer::Tensor *gate_tensor_ptr_; /**< ptr to gate weight (always FP32 resident) */

  virtual void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size,
    long long &sparsity_nz, long long &sparsity_total);

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
                    long long &activate_ns, long long &compute_ns,
                    long long &sparsity_nz, long long &sparsity_total);

  /** @brief Bump predicted experts to the MRU end so they survive eviction. */
  void touch_predicted(const std::vector<unsigned int> &predicted);

  /** @brief Deactivate (munmap) LRU-front experts until size <= cache_size. */
  void evict_experts(nntrainer::RunLayerContext &context);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__ */
