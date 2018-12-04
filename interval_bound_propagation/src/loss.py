# coding=utf-8
# Copyright 2018 The Interval Bound Propagation Authors.
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


ScalarMetrics = collections.namedtuple('ScalarMetrics', [
    'nominal_accuracy',
    'verified_accuracy',
    'attack_accuracy'])


ScalarLosses = collections.namedtuple('ScalarLosses', [
    'nominal_cross_entropy',
    'attack_cross_entropy',
    'verified_loss'])


class Losses(snt.AbstractModule):
  """Helper to compute our losses."""

  def __init__(self, predictor, specification=None, pgd_attack=None,
               interval_bounds_loss_type='xent',
               interval_bounds_hinge_margin=10.):
    super(Losses, self).__init__(name='losses')
    self._predictor = predictor
    self._specification = specification
    self._attack = pgd_attack
    if interval_bounds_loss_type not in ('xent', 'hinge'):
      raise ValueError('interval_bounds_loss_type must be either "xent" or '
                       '"hinge".')
    self._interval_bounds_loss_type = interval_bounds_loss_type
    self._interval_bounds_hinge_margin = interval_bounds_hinge_margin

  def _build(self, labels):
    # Cross-entropy.
    self._cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self._predictor.logits))
    # Accuracy.
    correct_examples = tf.equal(labels, tf.argmax(self._predictor.logits, 1))
    self._nominal_accuracy = tf.reduce_mean(
        tf.cast(correct_examples, tf.float32))

    # Interval bounds.
    if self._specification:
      bounds = self._specification(self._predictor.modules)
      v = tf.reduce_max(bounds, axis=1)
      self._interval_bounds_accuracy = tf.reduce_mean(
          tf.cast(v <= 0., tf.float32))
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
      else:
        assert self._interval_bounds_loss_type == 'hinge'
        self._verified_loss = tf.maximum(v, -self._interval_bounds_hinge_margin)

    else:
      self._verified_loss = tf.constant(0.)
      self._interval_bounds_accuracy = tf.constant(0.)

    # PGD attack.
    if self._attack:
      self._attack(labels)
      self._attack_accuracy = self._attack.accuracy
      self._attack_cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=self._attack.logits))
    else:
      self._attack_accuracy = tf.constant(0.)
      self._attack_cross_entropy = tf.constant(0.)

  @property
  def scalar_metrics(self):
    self._ensure_is_connected()
    return ScalarMetrics(self._nominal_accuracy,
                         self._interval_bounds_accuracy,
                         self._attack_accuracy)

  @property
  def scalar_losses(self):
    self._ensure_is_connected()
    return ScalarLosses(self._cross_entropy,
                        self._attack_cross_entropy,
                        self._verified_loss)
