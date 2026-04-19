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

#include "vw/common/random.h"

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
  float l2 = 0.0f;
  float dropout_rate = 0.0f;
  bool use_sigmoid = true;
  uint32_t hidden_size = 0;  // 0 = LRQ mode; >0 = NN two-tower mode with H hidden units per tower
  VW::workspace* all = nullptr;
  std::shared_ptr<VW::rand_state> random_state;

  // Buffers for NN tower mode (allocated when hidden_size > 0).
  std::vector<float> user_pre_act;
  std::vector<float> user_emb;
  std::vector<float> pos_pre_act;
  std::vector<float> pos_emb;
  std::vector<float> neg_pre_act;
  std::vector<float> neg_emb;

  // Dropout buffers (inverted dropout on embeddings during training).
  std::vector<float> user_drop;
  std::vector<float> pos_drop;
  std::vector<float> neg_drop;
  std::vector<uint8_t> drop_mask;  // 3*H: user|pos|neg masks
};

// ------------------------------------------------------------------ //
//  Fast tanh approximation (from nn.cc)                                //
// ------------------------------------------------------------------ //

static inline float fastpow2(float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = static_cast<int>(clipp);
  float z = clipp - w + offset;
  union
  {
    uint32_t i;
    float f;
  } v = {static_cast<uint32_t>(
      (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z))};
  return v.f;
}

static inline float fastexp(float p) { return fastpow2(1.442695040f * p); }
static inline float fasttanh(float p) { return -1.0f + 2.0f / (1.0f + fastexp(-2.0f * p)); }

// ------------------------------------------------------------------ //
//  NN Tower helpers                                                    //
// ------------------------------------------------------------------ //

static float dot_product(const float* a, const float* b, uint32_t H)
{
  float sum = 0.0f;
  for (uint32_t i = 0; i < H; ++i) { sum += a[i] * b[i]; }
  return sum;
}

// Score an example through a tower of H hidden units at weight offsets
// [offset..offset+H-1].  Pre-activation values stored in pre_act[],
// post-tanh activations in emb[].
static void score_tower(bpr_data& data, learner& base, VW::example& ex,
    uint32_t offset, uint32_t H, float* pre_act, float* emb, bool is_learn)
{
  float saved_label = ex.l.simple.label;
  float saved_weight = ex.weight;

  for (uint32_t i = 0; i < H; ++i)
  {
    base.predict(ex, offset + i);

    // Break symmetry: perturb away from saddle point at 0 (following nn.cc).
    if (is_learn && ex.pred.scalar == 0.0f)
    {
      float scale = 1.0f / std::sqrt(static_cast<float>(H));
      ex.l.simple.label = static_cast<float>(data.random_state->get_and_update_random() - 0.5) * scale;
      ex.weight = 1.0f;
      base.learn(ex, offset + i);
      ex.l.simple.label = saved_label;
      ex.weight = saved_weight;
      base.predict(ex, offset + i);
    }

    pre_act[i] = ex.pred.scalar;
    emb[i] = fasttanh(pre_act[i]);
  }
}

// Backpropagate through a tower: given gradient on post-tanh activations
// (grad_emb), compute gradient on pre-activations and inject into base
// learner via label manipulation.
static void backprop_tower(learner& base, VW::example& ex,
    uint32_t offset, uint32_t H, const float* pre_act, const float* emb, const float* grad_emb)
{
  for (uint32_t i = 0; i < H; ++i)
  {
    float tanh_deriv = 1.0f - emb[i] * emb[i];
    float grad_pre = grad_emb[i] * tanh_deriv;
    if (grad_pre == 0.0f) { continue; }

    // VW squared loss: first_derivative = 2*(pred - label).
    // To inject gradient g, set label = pred - g/2.
    ex.l.simple.label = pre_act[i] - 0.5f * grad_pre;
    ex.pred.scalar = pre_act[i];
    base.update(ex, offset + i);
  }
}

// ------------------------------------------------------------------ //
//  NN tower mode: predict a (user, item) pair                          //
// ------------------------------------------------------------------ //

static void predict_nn(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  if (examples.size() < 2) { return; }

  uint32_t H = data.hidden_size;
  VW::example& user_ex = *examples[0];
  VW::example& item_ex = *examples[1];

  score_tower(data, base, user_ex, 0, H, data.user_pre_act.data(), data.user_emb.data(), false);
  score_tower(data, base, item_ex, H, H, data.pos_pre_act.data(), data.pos_emb.data(), false);

  user_ex.pred.scalar = dot_product(data.user_emb.data(), data.pos_emb.data(), H);
}

// ------------------------------------------------------------------ //
//  NN tower mode: BPR pairwise update                                  //
// ------------------------------------------------------------------ //

static void learn_nn(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  if (examples.size() < 3)
  {
    predict_nn(data, base, examples);
    return;
  }

  uint32_t H = data.hidden_size;
  VW::example& user_ex = *examples[0];
  VW::example& pos_ex = *examples[1];
  VW::example& neg_ex = *examples[2];

  // Forward pass through towers.
  score_tower(data, base, user_ex, 0, H, data.user_pre_act.data(), data.user_emb.data(), true);
  score_tower(data, base, pos_ex, H, H, data.pos_pre_act.data(), data.pos_emb.data(), true);
  score_tower(data, base, neg_ex, H, H, data.neg_pre_act.data(), data.neg_emb.data(), true);

  // Apply inverted dropout to embeddings during training.
  const float* u_emb;
  const float* p_emb;
  const float* n_emb;
  bool use_dropout = data.dropout_rate > 0.0f;
  float drop_scale = use_dropout ? 1.0f / (1.0f - data.dropout_rate) : 1.0f;

  if (use_dropout)
  {
    for (uint32_t i = 0; i < H; ++i)
    {
      uint8_t um = data.random_state->get_and_update_random() >= data.dropout_rate ? 1 : 0;
      uint8_t pm = data.random_state->get_and_update_random() >= data.dropout_rate ? 1 : 0;
      uint8_t nm = data.random_state->get_and_update_random() >= data.dropout_rate ? 1 : 0;
      data.drop_mask[i] = um;
      data.drop_mask[H + i] = pm;
      data.drop_mask[2 * H + i] = nm;
      data.user_drop[i] = um ? data.user_emb[i] * drop_scale : 0.0f;
      data.pos_drop[i] = pm ? data.pos_emb[i] * drop_scale : 0.0f;
      data.neg_drop[i] = nm ? data.neg_emb[i] * drop_scale : 0.0f;
    }
    u_emb = data.user_drop.data();
    p_emb = data.pos_drop.data();
    n_emb = data.neg_drop.data();
  }
  else
  {
    u_emb = data.user_emb.data();
    p_emb = data.pos_emb.data();
    n_emb = data.neg_emb.data();
  }

  // Compute scores via dot product (using possibly-dropped embeddings).
  float s_pos = dot_product(u_emb, p_emb, H);
  float s_neg = dot_product(u_emb, n_emb, H);

  // Compute BPR gradient.
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
      user_ex.loss = 0.0f;
      user_ex.pred.scalar = s_pos;
      return;
    }
  }

  // Backprop through dot product to get gradients on embeddings.
  std::vector<float> user_grad(H), pos_grad(H), neg_grad(H);
  for (uint32_t i = 0; i < H; ++i)
  {
    float du = grad_pos * p_emb[i] + grad_neg * n_emb[i];
    float dp = grad_pos * u_emb[i];
    float dn = grad_neg * u_emb[i];

    if (use_dropout)
    {
      // Chain rule through dropout: dL/d(emb) = dL/d(dropped) * mask * scale.
      user_grad[i] = data.drop_mask[i] ? du * drop_scale : 0.0f;
      pos_grad[i] = data.drop_mask[H + i] ? dp * drop_scale : 0.0f;
      neg_grad[i] = data.drop_mask[2 * H + i] ? dn * drop_scale : 0.0f;
    }
    else
    {
      user_grad[i] = du;
      pos_grad[i] = dp;
      neg_grad[i] = dn;
    }
  }

  // L2 regularization on embedding norms: dL/d(emb[i]) += l2 * emb[i].
  if (data.l2 > 0.0f)
  {
    for (uint32_t i = 0; i < H; ++i)
    {
      user_grad[i] += data.l2 * data.user_emb[i];
      pos_grad[i] += data.l2 * data.pos_emb[i];
      neg_grad[i] += data.l2 * data.neg_emb[i];
    }
  }

  // Save labels/weights.
  float saved_user_label = user_ex.l.simple.label;
  float saved_pos_label = pos_ex.l.simple.label;
  float saved_neg_label = neg_ex.l.simple.label;
  float saved_user_weight = user_ex.weight;
  float saved_pos_weight = pos_ex.weight;
  float saved_neg_weight = neg_ex.weight;

  user_ex.weight = 1.0f;
  pos_ex.weight = 1.0f;
  neg_ex.weight = 1.0f;

  // Backprop through each tower (using original pre_act/emb for tanh derivative).
  backprop_tower(base, user_ex, 0, H, data.user_pre_act.data(), data.user_emb.data(), user_grad.data());
  backprop_tower(base, pos_ex, H, H, data.pos_pre_act.data(), data.pos_emb.data(), pos_grad.data());
  backprop_tower(base, neg_ex, H, H, data.neg_pre_act.data(), data.neg_emb.data(), neg_grad.data());

  // Restore labels/weights.
  user_ex.l.simple.label = saved_user_label;
  pos_ex.l.simple.label = saved_pos_label;
  neg_ex.l.simple.label = saved_neg_label;
  user_ex.weight = saved_user_weight;
  pos_ex.weight = saved_pos_weight;
  neg_ex.weight = saved_neg_weight;

  // Compute BPR loss for progress reporting.
  {
    float diff = s_pos - s_neg;
    diff = std::max(-30.0f, std::min(30.0f, diff));
    float bpr_loss;
    if (data.use_sigmoid) { bpr_loss = std::log(1.0f + std::exp(-diff)); }
    else { bpr_loss = std::max(0.0f, -diff + data.margin); }
    if (data.l2 > 0.0f)
    {
      float emb_norm_sq = 0.0f;
      for (uint32_t i = 0; i < H; ++i)
      {
        emb_norm_sq += data.user_emb[i] * data.user_emb[i]
            + data.pos_emb[i] * data.pos_emb[i]
            + data.neg_emb[i] * data.neg_emb[i];
      }
      bpr_loss += 0.5f * data.l2 * emb_norm_sq;
    }
    user_ex.loss = bpr_loss;
  }

  user_ex.pred.scalar = s_pos;
}

// ------------------------------------------------------------------ //
//  Learn: BPR pairwise update (LRQ mode)                               //
// ------------------------------------------------------------------ //

void learn(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  if (data.hidden_size > 0) { learn_nn(data, base, examples); return; }
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
    // BPR-sigmoid: L = -ln σ(s_pos - s_neg)
    //
    //   Let d = s_pos - s_neg.
    //   dL/ds_pos = σ(d) - 1 = -σ(-d)   (always <= 0, pushes s_pos up)
    //   dL/ds_neg = 1 - σ(d) =  σ(-d)   (always >= 0, pushes s_neg down)
    //
    //   When d >> 0 (correct ranking): gradients → 0 (no update needed)
    //   When d << 0 (wrong ranking):   gradients → -1 / +1 (max correction)
    float diff = s_pos - s_neg;
    diff = std::max(-30.0f, std::min(30.0f, diff));  // clamp to avoid exp overflow
    float sig = 1.0f / (1.0f + std::exp(-diff));     // σ(s_pos - s_neg)

    grad_pos = sig - 1.0f;   // always <= 0
    grad_neg = 1.0f - sig;   // always >= 0
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
      anchor_ex.loss = 0.0f;
      anchor_ex.pred.scalar = s_pos;
      return;
    }
  }

  // L2 regularization: penalise absolute prediction magnitude to prevent
  // unbounded score drift (BPR only constrains the *difference*).
  // dL2/ds = l2 * s   for each score independently.
  if (data.l2 > 0.0f)
  {
    grad_pos += data.l2 * s_pos;
    grad_neg += data.l2 * s_neg;
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
  float orig_pos_weight = pos_ex.weight;
  float orig_neg_weight = neg_ex.weight;

  // BPR pairs are uniformly weighted; importance comes from sampling frequency.
  pos_ex.weight = 1.0f;
  neg_ex.weight = 1.0f;

  // VW's squared loss first_derivative = 2*(pred - label).
  // To inject gradient g, set label = pred - g/2 so that 2*(pred - label) = g.
  pos_ex.l.simple.label = s_pos - 0.5f * grad_pos;
  neg_ex.l.simple.label = s_neg - 0.5f * grad_neg;

  // 5. Learn: base learner applies SGD with our injected gradients.
  base.learn(pos_ex);
  base.learn(neg_ex);

  // 6. Restore original labels, weights, and un-merge anchor features.
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

    // Include L2 penalty in reported loss: 0.5 * l2 * (s_pos^2 + s_neg^2)
    if (data.l2 > 0.0f) { bpr_loss += 0.5f * data.l2 * (s_pos * s_pos + s_neg * s_neg); }
    anchor_ex.loss = bpr_loss;
  }

  // Store the positive score as the prediction for this multi_ex.
  anchor_ex.pred.scalar = s_pos;
}

// ------------------------------------------------------------------ //
//  Predict: score a (user, item) pair                                  //
// ------------------------------------------------------------------ //

void predict(bpr_data& data, learner& base, VW::multi_ex& examples)
{
  if (data.hidden_size > 0) { predict_nn(data, base, examples); return; }

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
      // Show BPR loss as "label" column and positive score as "predict".
      sd.print_update(*all.output_runtime.trace_message, all.passes_config.holdout_set_off,
          all.passes_config.current_pass, ec.loss, ec.pred.scalar, ec.get_num_features());
    }
  }
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
  float bpr_l2 = 0.0f;
  float bpr_dropout = 0.0f;
  bool bpr_sigmoid = true;
  uint32_t bpr_hidden = 0;

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
               .help("L2 regularization (scores in LRQ mode, embedding norms in NN mode)"))
      .add(make_option("bpr_dropout", bpr_dropout)
               .default_value(0.0f)
               .help("Dropout rate for NN two-tower embeddings (0.0 = disabled, e.g. 0.5)"))
      .add(make_option("bpr_hidden", bpr_hidden)
               .keep()
               .default_value(0u)
               .help("Hidden size for NN two-tower mode (0 = use base/LRQ, >0 = two-tower NN with tanh)"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  auto data = VW::make_unique<bpr_data>();
  data->margin = bpr_margin;
  data->l2 = bpr_l2;
  data->dropout_rate = bpr_dropout;
  data->use_sigmoid = bpr_sigmoid;
  data->hidden_size = bpr_hidden;
  data->all = stack_builder.get_all_pointer();
  data->random_state = stack_builder.get_all_pointer()->get_random_state();

  if (bpr_hidden > 0)
  {
    data->user_pre_act.resize(bpr_hidden);
    data->user_emb.resize(bpr_hidden);
    data->pos_pre_act.resize(bpr_hidden);
    data->pos_emb.resize(bpr_hidden);
    data->neg_pre_act.resize(bpr_hidden);
    data->neg_emb.resize(bpr_hidden);
    if (bpr_dropout > 0.0f)
    {
      data->user_drop.resize(bpr_hidden);
      data->pos_drop.resize(bpr_hidden);
      data->neg_drop.resize(bpr_hidden);
      data->drop_mask.resize(3 * bpr_hidden);
    }
  }

  // feature_width: in NN tower mode, 2*H weight sets (H per tower).
  // In LRQ mode, 1 (LRQ manages its own width).
  size_t feature_width = (bpr_hidden > 0) ? 2 * static_cast<size_t>(bpr_hidden) : 1;
  auto base = require_singleline(stack_builder.setup_base_learner(feature_width));

  // BPR is multiline (receives triplets), delegates to singleline base.
  auto l = make_reduction_learner(std::move(data), base, learn, predict,
      stack_builder.get_setupfn_name(bpr_setup))
               .set_feature_width(feature_width)
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
