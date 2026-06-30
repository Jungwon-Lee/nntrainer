// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_prefetch_layer.h
 * @date   26 June 2026
 * @brief  Pass-through layer that fires background expert activation before
 *         attention runs, overlapping page-in latency with attention compute.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_MOE_PREFETCH_LAYER_H__
#define __SMALLTHINKER_MOE_PREFETCH_LAYER_H__
#ifdef __cplusplus

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <smallthinker_moe_layer.h>
#include <tuple>

namespace causallm {

namespace props {

/**
 * @brief MoELayerId — the decoder layer index used to look up the paired
 *        SmallThinkerCachedSlimMoELayer in the prefetch registry.
 *        Layer indices start at 0, so cannot use PositiveIntegerProperty.
 */
class MoELayerId : public nntrainer::Property<unsigned int> {
public:
  MoELayerId(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "moe_layer_id";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class SmallThinkerMoEPrefetchLayer
 * @brief Inserted at the start of each SmallThinker decoder block. Consumes
 *        the block's pre-attention input, computes router logits using the
 *        paired compute-node's gate weight (shared via global registry), fires
 *        background activate() tasks for the predicted expert set, then passes
 *        the input through unchanged as its output.
 *
 *        This shifts expert page-in (mmap + madvise WILLNEED on Android) to
 *        overlap with attention, hiding memory-I/O latency on RAM-constrained
 *        edge devices.
 *
 *        On the first token (cold start) the gate Tensor pointer is not yet
 *        populated, so the layer is a no-op that pass — no correctness effect.
 *        Set NNTR_MOE_PREFETCH_THREADS=0 to disable prefetch entirely.
 */
class SmallThinkerMoEPrefetchLayer : public nntrainer::LayerImpl {
public:
  SmallThinkerMoEPrefetchLayer();
  ~SmallThinkerMoEPrefetchLayer() = default;

  SmallThinkerMoEPrefetchLayer(SmallThinkerMoEPrefetchLayer &&) noexcept =
    default;
  SmallThinkerMoEPrefetchLayer &
  operator=(SmallThinkerMoEPrefetchLayer &&) = default;

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

  const std::string getType() const override {
    return SmallThinkerMoEPrefetchLayer::type;
  }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "smallthinker_moe_prefetch";

private:
  unsigned int num_experts_;
  unsigned int topk_;
  bool router_apply_softmax_;
  int moe_layer_id_;

  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             props::MoERouterApplySoftmax, props::MoELayerId>
    prefetch_props_;

  unsigned int router_logits_idx_;

  // Fire prefetch tasks for the given token window using the shared registry.
  void firePrefetch(nntrainer::Tensor &router_input,
                    unsigned int total_tokens) const;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_PREFETCH_LAYER_H__ */
