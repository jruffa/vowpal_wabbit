// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
//
// BPR (Bayesian Personalized Ranking) reduction for VowpalWabbit.
//
// Implements pairwise ranking loss for implicit feedback. Sits as a
// multiline reduction on top of a singleline base learner (typically --lrq).
//
// Data format (multi-line example, blank line delimited):
//   0 |U device_features
//   1 [importance_weight] |I positive_item_features
//   -1 |I negative_item_features
//
// The first example in each multi_ex is the anchor (user/device context).
// The second is the positive item (label > 0).
// The third is the negative item (label < 0).
//
// Loss functions:
//   BPR-sigmoid (default):  L = -ln sigma(s_pos - s_neg)
//   Hinge:                  L = max(0, s_neg - s_pos + margin)
//
// Gradients are injected into the base learner via label manipulation.
// For squared loss base: update = eta * (pred - label), so setting
// label = pred - desired_gradient yields the correct update direction.

#include "vw/core/reductions/bpr.h"

#include "vw/config/options.h"
#include "vw/core/example.h"
#include "vw/core/learner.h"
#include "vw/core/setup_base.h"
#include "vw/core/shared_data.h"
#include "vw/core/simple_label.h"
#include "vw/core/vw.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace VW::LEARNER;
using namespace VW::config;

namespace
{

// ------------------------------------------------------------------ //
//  Reduction state                                                     //
// ------------------------------------------------------------------ //

struct bpr_data
{
  float margin = 1.0f;
  bool use_sigmoid = true;
  VW::workspace* all = nullptr;
};

// ------------------------------------------------------------------ //
//  Learn: BPR pairwise update                                          //
// ------------------------------------------------------------------ //

void learn(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  // Validate multi-line example structure:
  //   examples[0] = anchor (user/device features, |U), label = 0
  //   examples[1] = positive item (|I), label > 0
  //   examples[2] = negative item (|I), label < 0
  //
  // At prediction time we may get only 2 lines (anchor + item).
  if (examples.size() < 3)
  {
    if (examples.size() >= 2)
    {
      // Merge anchor features into item and predict.
      VW::example& anchor_ex = *examples[0];
      VW::example& item_ex = *examples[1];
      VW::details::append_example_namespaces_from_example(item_ex, anchor_ex);
      base.predict(item_ex);
      anchor_ex.pred.scalar = item_ex.pred.scalar;
      VW::details::truncate_example_namespaces_from_example(item_ex, anchor_ex);
    }
    return;
  }

  VW::example& anchor_ex = *examples[0];
  VW::example& pos_ex = *examples[1];
  VW::example& neg_ex = *examples[2];

  // 1. Merge anchor features into both positive and negative examples.
  VW::details::append_example_namespaces_from_example(pos_ex, anchor_ex);
  VW::details::append_example_namespaces_from_example(neg_ex, anchor_ex);

  // 2. Score both items via the base learner.
  base.predict(pos_ex);
  float s_pos = pos_ex.pred.scalar;

  base.predict(neg_ex);
  float s_neg = neg_ex.pred.scalar;

  // 3. Compute BPR gradient.
  float grad_pos, grad_neg;

  if (data.use_sigmoid)
  {
    // BPR-sigmoid: L = -ln sigma(s_pos - s_neg)
    //   dL/ds_pos = sigma(s_neg - s_pos) - 1   (always <= 0, pushes s_pos up)
    //   dL/ds_neg = sigma(s_neg - s_pos)        (always >= 0, pushes s_neg down)
    float diff = s_neg - s_pos;
    diff = std::max(-30.0f, std::min(30.0f, diff));  // clamp to avoid exp overflow
    float sig = 1.0f / (1.0f + std::exp(-diff));

    grad_pos = sig - 1.0f;
    grad_neg = sig;
  }
  else
  {
    // Hinge: L = max(0, s_neg - s_pos + margin)
    float violation = s_neg - s_pos + data.margin;
    if (violation > 0.0f)
    {
      grad_pos = -1.0f;
      grad_neg = 1.0f;
    }
    else
    {
      // Margin satisfied — no update needed.
      VW::details::truncate_example_namespaces_from_example(pos_ex, anchor_ex);
      VW::details::truncate_example_namespaces_from_example(neg_ex, anchor_ex);
      anchor_ex.pred.scalar = s_pos;
      return;
    }
  }

  // 4. Inject gradients via label manipulation.
  //
  //    For squared loss the base learner computes:
  //      update = learning_rate * (prediction - label)
  //
  //    We want update = learning_rate * grad, so we set:
  //      label = prediction - grad
  //
  //    This makes (prediction - label) = grad, which is our desired gradient.

  float orig_pos_label = pos_ex.l.simple.label;
  float orig_neg_label = neg_ex.l.simple.label;

  pos_ex.l.simple.label = s_pos - grad_pos;
  neg_ex.l.simple.label = s_neg - grad_neg;

  // 5. Learn: base learner applies SGD with our injected gradients.
  base.learn(pos_ex);
  base.learn(neg_ex);

  // 6. Restore original labels and un-merge anchor features.
  pos_ex.l.simple.label = orig_pos_label;
  neg_ex.l.simple.label = orig_neg_label;

  VW::details::truncate_example_namespaces_from_example(pos_ex, anchor_ex);
  VW::details::truncate_example_namespaces_from_example(neg_ex, anchor_ex);

  // Store the positive score as the prediction for this multi_ex.
  anchor_ex.pred.scalar = s_pos;
}

// ------------------------------------------------------------------ //
//  Predict: score a (user, item) pair                                  //
// ------------------------------------------------------------------ //

void predict(bpr_data& /*data*/, learner& base, VW::multi_ex& examples)
{
  if (examples.size() < 2) { return; }

  VW::example& anchor_ex = *examples[0];
  VW::example& item_ex = *examples[1];

  VW::details::append_example_namespaces_from_example(item_ex, anchor_ex);

  base.predict(item_ex);
  float score = item_ex.pred.scalar;

  VW::details::truncate_example_namespaces_from_example(item_ex, anchor_ex);

  anchor_ex.pred.scalar = score;
}

// ------------------------------------------------------------------ //
//  Example lifecycle hooks (required for multiline→singleline bridge)  //
// ------------------------------------------------------------------ //

void update_stats_bpr(const VW::workspace& all, VW::shared_data& sd, const bpr_data& /*data*/,
    const VW::multi_ex& ec_seq, VW::io::logger& logger)
{
  if (!ec_seq.empty()) { VW::details::update_stats_simple_label(all, sd, *ec_seq[0], logger); }
}

void output_example_prediction_bpr(
    VW::workspace& all, const bpr_data& /*data*/, const VW::multi_ex& ec_seq, VW::io::logger& logger)
{
  if (!ec_seq.empty()) { VW::details::output_example_prediction_simple_label(all, *ec_seq[0], logger); }
}

void print_update_bpr(VW::workspace& all, VW::shared_data& sd, const bpr_data& /*data*/,
    const VW::multi_ex& ec_seq, VW::io::logger& logger)
{
  if (!ec_seq.empty()) { VW::details::print_update_simple_label(all, sd, *ec_seq[0], logger); }
}

}  // anonymous namespace

// ------------------------------------------------------------------ //
//  Setup: parse options, build reduction                                //
// ------------------------------------------------------------------ //

std::shared_ptr<VW::LEARNER::learner> VW::reductions::bpr_setup(VW::setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();

  bool bpr_enabled = false;
  float bpr_margin = 1.0f;
  bool bpr_sigmoid = true;

  option_group_definition new_options("[Reduction] BPR - Bayesian Personalized Ranking");
  new_options
      .add(make_option("bpr", bpr_enabled)
               .keep()
               .necessary()
               .help("Enable BPR pairwise ranking reduction for implicit feedback"))
      .add(make_option("bpr_margin", bpr_margin)
               .default_value(1.0f)
               .help("Margin for hinge loss variant (default: 1.0)"))
      .add(make_option("bpr_sigmoid", bpr_sigmoid)
               .default_value(true)
               .help("Use BPR-sigmoid loss instead of hinge (default: true)"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  auto data = VW::make_unique<bpr_data>();
  data->margin = bpr_margin;
  data->use_sigmoid = bpr_sigmoid;
  data->all = stack_builder.get_all_pointer();

  // The base learner is singleline (e.g., --lrq produces scalar predictions).
  auto base = require_singleline(stack_builder.setup_base_learner());

  // BPR is multiline (receives triplets), delegates to singleline base.
  auto l = make_reduction_learner(std::move(data), base, learn, predict,
      stack_builder.get_setupfn_name(bpr_setup))
               .set_input_label_type(VW::label_type_t::SIMPLE)
               .set_output_label_type(VW::label_type_t::SIMPLE)
               .set_input_prediction_type(VW::prediction_type_t::SCALAR)
               .set_output_prediction_type(VW::prediction_type_t::SCALAR)
               .set_update_stats(update_stats_bpr)
               .set_output_example_prediction(output_example_prediction_bpr)
               .set_print_update(print_update_bpr)
               .build();

  return l;
}
