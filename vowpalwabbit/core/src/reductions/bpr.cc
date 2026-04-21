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
  float l2 = 0.0f;
  bool use_sigmoid = true;
  VW::workspace* all = nullptr;
};

// ------------------------------------------------------------------ //
//  Learn: BPR pairwise update                                          //
// ------------------------------------------------------------------ //

void learn(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  if (examples.size() < 3)
  {
    if (examples.size() >= 2)
    {
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
    float diff = s_pos - s_neg;
    diff = std::max(-30.0f, std::min(30.0f, diff));
    float sig = 1.0f / (1.0f + std::exp(-diff));

    grad_pos = sig - 1.0f;
    grad_neg = 1.0f - sig;
  }
  else
  {
    float violation = s_neg - s_pos + data.margin;
    if (violation > 0.0f)
    {
      grad_pos = -1.0f;
      grad_neg = 1.0f;
    }
    else
    {
      VW::details::truncate_example_namespaces_from_example(pos_ex, anchor_ex);
      VW::details::truncate_example_namespaces_from_example(neg_ex, anchor_ex);
      anchor_ex.loss = 0.0f;
      anchor_ex.pred.scalar = s_pos;
      return;
    }
  }

  // L2 regularization on score magnitude.
  if (data.l2 > 0.0f)
  {
    grad_pos += data.l2 * s_pos;
    grad_neg += data.l2 * s_neg;
  }

  // 4. Inject gradients via label manipulation.
  //    Squared loss: first_derivative = 2*(pred - label).
  //    Set label = pred - grad/2 so that 2*(pred - label) = grad.
  float orig_pos_label = pos_ex.l.simple.label;
  float orig_neg_label = neg_ex.l.simple.label;
  float orig_pos_weight = pos_ex.weight;
  float orig_neg_weight = neg_ex.weight;

  pos_ex.weight = 1.0f;
  neg_ex.weight = 1.0f;

  pos_ex.l.simple.label = s_pos - 0.5f * grad_pos;
  neg_ex.l.simple.label = s_neg - 0.5f * grad_neg;

  // 5. Learn.
  base.learn(pos_ex);
  base.learn(neg_ex);

  // 6. Restore and un-merge.
  pos_ex.l.simple.label = orig_pos_label;
  neg_ex.l.simple.label = orig_neg_label;
  pos_ex.weight = orig_pos_weight;
  neg_ex.weight = orig_neg_weight;

  VW::details::truncate_example_namespaces_from_example(pos_ex, anchor_ex);
  VW::details::truncate_example_namespaces_from_example(neg_ex, anchor_ex);

  // Compute BPR loss for progress reporting.
  {
    float diff = s_pos - s_neg;
    diff = std::max(-30.0f, std::min(30.0f, diff));
    float bpr_loss;
    if (data.use_sigmoid) { bpr_loss = std::log(1.0f + std::exp(-diff)); }
    else { bpr_loss = std::max(0.0f, -diff + data.margin); }
    if (data.l2 > 0.0f) { bpr_loss += 0.5f * data.l2 * (s_pos * s_pos + s_neg * s_neg); }
    anchor_ex.loss = bpr_loss;
  }

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
//  Example lifecycle hooks                                             //
// ------------------------------------------------------------------ //

void update_stats_bpr(const VW::workspace& /*all*/, VW::shared_data& sd, const bpr_data& /*data*/,
    const VW::multi_ex& ec_seq, VW::io::logger& /*logger*/)
{
  if (!ec_seq.empty())
  {
    const auto& ec = *ec_seq[0];
    sd.update(ec.test_only, true, ec.loss, ec.weight, ec.get_num_features());
  }
}

void output_example_prediction_bpr(
    VW::workspace& all, const bpr_data& /*data*/, const VW::multi_ex& ec_seq, VW::io::logger& logger)
{
  if (!ec_seq.empty()) { VW::details::output_example_prediction_simple_label(all, *ec_seq[0], logger); }
}

void print_update_bpr(VW::workspace& all, VW::shared_data& sd, const bpr_data& /*data*/,
    const VW::multi_ex& ec_seq, VW::io::logger& /*logger*/)
{
  if (!ec_seq.empty())
  {
    const auto& ec = *ec_seq[0];
    const bool should_print =
        sd.weighted_examples() >= sd.dump_interval && !all.output_config.quiet && !all.reduction_state.bfgs;
    if (should_print)
    {
      sd.print_update(*all.output_runtime.trace_message, all.passes_config.holdout_set_off,
          all.passes_config.current_pass, ec.loss, ec.pred.scalar, ec.get_num_features());
    }
  }
}

}  // anonymous namespace

// ------------------------------------------------------------------ //
//  Setup                                                               //
// ------------------------------------------------------------------ //

std::shared_ptr<VW::LEARNER::learner> VW::reductions::bpr_setup(VW::setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();

  bool bpr_enabled = false;
  float bpr_margin = 1.0f;
  float bpr_l2 = 0.0f;
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
               .help("Use BPR-sigmoid loss instead of hinge (default: true)"))
      .add(make_option("bpr_l2", bpr_l2)
               .default_value(0.0f)
               .help("L2 regularization on score magnitude"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  auto data = VW::make_unique<bpr_data>();
  data->margin = bpr_margin;
  data->l2 = bpr_l2;
  data->use_sigmoid = bpr_sigmoid;
  data->all = stack_builder.get_all_pointer();

  auto base = require_singleline(stack_builder.setup_base_learner(1));

  auto l = make_reduction_learner(std::move(data), base, learn, predict,
      stack_builder.get_setupfn_name(bpr_setup))
               .set_feature_width(1)
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
