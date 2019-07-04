# coding=utf-8
# Copyright 2019 The Interval Bound Propagation Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper to keep track of the different losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


# Used to pick the least violated specification.
_BIG_NUMBER = 1e25


ScalarMetrics = collections.namedtuple('ScalarMetrics', [
    'nominal_accuracy',
    'verified_accuracy',
    'attack_accuracy',
    'attack_success'])


ScalarLosses = collections.namedtuple('ScalarLosses', [
    'nominal_cross_entropy',
    'attack_cross_entropy',
    'verified_loss'])


class Losses(snt.AbstractModule):
  """Helper to compute our losses."""

  def __init__(self, predictor, specification=None, pgd_attack=None,
               interval_bounds_loss_type='xent',
               interval_bounds_hinge_margin=10.,
               label_smoothing=0.,
               pgd_attack_use_trades=False):
    super(Losses, self).__init__(name='losses')
    self._predictor = predictor
    self._specification = specification
    self._attack = pgd_attack
    # Loss type can be any combination of:
    #   xent: cross-entropy loss
    #   hinge: hinge loss
    #   softplus: softplus loss
    # with
    #   all: using all specifications.
    #   most: using only the specification that is the most violated.
    #   least: using only the specification that is the least violated.
    #   random_n: using a random subset of the specifications.
    # E.g.: "xent_max" or "hinge_random_3".
    tokens = interval_bounds_loss_type.split('_', 1)
    if len(tokens) == 1:
      loss_type, loss_mode = tokens[0], 'all'
    else:
      loss_type, loss_mode = tokens
      if loss_mode.startswith('random'):
        loss_mode, num_samples = loss_mode.split('_', 1)
        self._interval_bounds_loss_n = int(num_samples)
    if loss_type not in ('xent', 'hinge', 'softplus'):
      raise ValueError('interval_bounds_loss_type must be either "xent", '
                       '"hinge" or "softplus".')
    if loss_mode not in ('all', 'most', 'random', 'least'):
      raise ValueError('interval_bounds_loss_type must be followed by either '
                       '"all", "most", "random_N" or "least".')
    self._interval_bounds_loss_type = loss_type
    self._interval_bounds_loss_mode = loss_mode
    self._interval_bounds_hinge_margin = interval_bounds_hinge_margin
    self._label_smoothing = label_smoothing
    self._pgd_attack_use_trades = pgd_attack_use_trades

  def _build(self, labels):
    # Cross-entropy.
    nominal_logits = self._predictor.logits
    if self._label_smoothing > 0:
      num_classes = nominal_logits.shape[1].value
      one_hot_labels = tf.one_hot(labels, num_classes)
      smooth_positives = 1. - self._label_smoothing
      smooth_negatives = self._label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
      nominal_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=one_hot_labels, logits=nominal_logits)
    else:
      nominal_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=nominal_logits)
    self._cross_entropy = tf.reduce_mean(nominal_cross_entropy)
    # Accuracy.
    nominal_correct_examples = tf.equal(labels, tf.argmax(nominal_logits, 1))
    self._nominal_accuracy = tf.reduce_mean(
        tf.cast(nominal_correct_examples, tf.float32))

    # Interval bounds.
    if self._specification:
      bounds = self._specification(self._predictor.modules)
      v = tf.reduce_max(bounds, axis=1)
      self._interval_bounds_accuracy = tf.reduce_mean(
          tf.cast(v <= 0., tf.float32))
      # Select specifications.
      if self._interval_bounds_loss_mode == 'all':
        pass  # Keep bounds the way it is.
      elif self._interval_bounds_loss_mode == 'most':
        bounds = tf.reduce_max(bounds, axis=1, keepdims=True)
      elif self._interval_bounds_loss_mode == 'random':
        idx = tf.random.uniform(
            [tf.shape(bounds)[0], self._interval_bounds_loss_n],
            0, tf.shape(bounds)[1], dtype=tf.int32)
        bounds = tf.batch_gather(bounds, idx)
      else:
        assert self._interval_bounds_loss_mode == 'least'
        # This picks the least violated contraint.
        mask = tf.cast(bounds < 0., tf.float32)
        smallest_violation = tf.reduce_min(
            bounds + mask * _BIG_NUMBER, axis=1, keepdims=True)
        has_violations = tf.less(
            tf.reduce_sum(mask, axis=1, keepdims=True) + .5,
            tf.cast(tf.shape(bounds)[1], tf.float32))
        largest_bounds = tf.reduce_max(bounds, axis=1, keepdims=True)
        bounds = tf.where(has_violations, smallest_violation, largest_bounds)

      if self._interval_bounds_loss_type == 'xent':
        v = tf.concat(
            [bounds, tf.zeros([tf.shape(bounds)[0], 1], dtype=bounds.dtype)],
            axis=1)
        l = tf.concat(
            [tf.zeros_like(bounds),
             tf.ones([tf.shape(bounds)[0], 1], dtype=bounds.dtype)],
            axis=1)
        self._verified_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(l), logits=v))
      elif self._interval_bounds_loss_type == 'softplus':
        self._verified_loss = tf.reduce_mean(
            tf.nn.softplus(bounds + self._interval_bounds_hinge_margin))
      else:
        assert self._interval_bounds_loss_type == 'hinge'
        self._verified_loss = tf.reduce_mean(
            tf.maximum(bounds, -self._interval_bounds_hinge_margin))
    else:
      self._verified_loss = tf.constant(0.)
      self._interval_bounds_accuracy = tf.constant(0.)

    # PGD attack.
    if self._attack:
      if not isinstance(self._predictor.inputs, tf.Tensor):
        raise ValueError('Multiple inputs is not supported.')
      self._attack(self._predictor.inputs, labels)
      correct_examples = tf.equal(labels, tf.argmax(self._attack.logits, 1))
      self._attack_accuracy = tf.reduce_mean(
          tf.cast(correct_examples, tf.float32))
      self._attack_success = tf.reduce_mean(
          tf.cast(self._attack.success, tf.float32))
      if self._pgd_attack_use_trades:
        # The variable is misnamed in this case.
        nominal_logits = tf.stop_gradient(nominal_logits)
        attack_cross_entropy = tfp.distributions.kl_divergence(
            tfp.distributions.Categorical(logits=nominal_logits),
            tfp.distributions.Categorical(logits=self._attack.logits))
      else:
        if self._label_smoothing > 0:
          attack_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=one_hot_labels, logits=self._attack.logits)
        else:
          attack_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=self._attack.logits)
      self._attack_cross_entropy = tf.reduce_mean(attack_cross_entropy)
    else:
      self._attack_accuracy = tf.constant(0.)
      self._attack_success = tf.constant(1.)
      self._attack_cross_entropy = tf.constant(0.)

  @property
  def scalar_metrics(self):
    self._ensure_is_connected()
    return ScalarMetrics(self._nominal_accuracy,
                         self._interval_bounds_accuracy,
                         self._attack_accuracy,
                         self._attack_success)

  @property
  def scalar_losses(self):
    self._ensure_is_connected()
    return ScalarLosses(self._cross_entropy,
                        self._attack_cross_entropy,
                        self._verified_loss)
