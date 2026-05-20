/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	causallm_common_properties.h
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __CAUSALLM_COMMON_PROPERTIES_H__
#define __CAUSALLM_COMMON_PROPERTIES_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <common_properties.h>
#include <tensor.h>
#include <utility>

namespace causallm {

namespace props {

/**
 * @brief MoE activation type
 */
class MoEActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "moe_activation";
};
/**
 * @brief NumExperts,  Number of experts property
 */
class NumExperts : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "num_experts"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

/**
 * @brief NumExpertsPerToken,  Number of experts per token property
 */
class NumExpertsPerToken : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "num_experts_per_token";                 /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class FeatureSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "feature_size";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief Square base grid size for resized 2D position embeddings.
 */
class BaseGridSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "base_grid_size";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief Spatial downsample factor for vision projector reshaping.
 */
class DownsampleFactor : public nntrainer::PositiveIntegerProperty {
public:
  DownsampleFactor(unsigned int value = 2) { set(value); }
  static constexpr const char *key =
    "downsample_factor";                     /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief Token id used as an image-feature placeholder in input_ids.
 */
class ImageTokenId : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "image_token_id";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief Additive attention value for masked entries.
 */
class MaskValue : public nntrainer::Property<float> {
public:
  MaskValue(float value = -1.0e10f) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "mask_value"; /**< unique key */
  using prop_tag = nntrainer::float_prop_tag;      /**< property type */
};

/**
 * @brief RMS_NORM_GAMMA_INIT Initialization Enumeration Information
 *
 */
WIN_EXPORT class RMS_NORM_GAMMA_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a CUSTOM_RMS_NORM_GAMMA_INIT object
   */
  WIN_EXPORT RMS_NORM_GAMMA_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
}; // namespace props

WIN_EXPORT enum RMSParams { gamma };

} // namespace causallm

#endif
