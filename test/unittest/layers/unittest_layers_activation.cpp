// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_pooling.cpp
 * @date 7 July 2021
 * @brief Activation Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <activation_layer.h>
#include <layer_context.h>
#include <layers_common_tests.h>

auto semantic_activation_relu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=relu"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_swish = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=swish"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_gelu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=gelu"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_sigmoid = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=sigmoid"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_softmax = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=softmax"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_tanh = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=tanh"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_none = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=none"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(
  Activation, LayerSemantics,
  ::testing::Values(semantic_activation_relu, semantic_activation_swish,
                    semantic_activation_gelu, semantic_activation_sigmoid,
                    semantic_activation_softmax, semantic_activation_tanh,
                    semantic_activation_none));

TEST(ActivationUpdateTensorsByInputDimensions,
     updates_input_output_height_only) {
  nntrainer::ActivationLayer acti;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs = {nntrainer::Var_Grad(
    nntrainer::TensorDim({1, 1, 4, 8}), nntrainer::Initializer::NONE, true,
    false, "activation_input")};
  std::vector<nntrainer::Var_Grad> outputs = {nntrainer::Var_Grad(
    nntrainer::TensorDim({1, 1, 4, 8}), nntrainer::Initializer::NONE, true,
    false, "activation_output")};
  std::vector<nntrainer::Var_Grad> tensors;

  std::vector<nntrainer::Weight *> weights_view;
  std::vector<nntrainer::Var_Grad *> inputs_view = {&inputs[0]};
  std::vector<nntrainer::Var_Grad *> outputs_view = {&outputs[0]};
  std::vector<nntrainer::Var_Grad *> tensors_view;

  nntrainer::RunLayerContext context("activation_update_dims_test", true, 0.0f,
                                     false, 1.0f, nullptr, false, weights_view,
                                     inputs_view, outputs_view, tensors_view);

  std::vector<nntrainer::TensorDim> dims = {nntrainer::TensorDim({1, 1, 6, 8})};

  EXPECT_NO_THROW(acti.updateTensorsByInputDimensions(context, dims));

  EXPECT_EQ(context.getInput(0).getDim().height(), 6u);
  EXPECT_EQ(context.getInput(0).getDim().width(), 8u);
  EXPECT_EQ(context.getOutput(0).getDim().height(), 6u);
  EXPECT_EQ(context.getOutput(0).getDim().width(), 8u);
}
