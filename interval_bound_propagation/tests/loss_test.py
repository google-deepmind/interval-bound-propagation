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

"""Tests for loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


class FixedNN(snt.AbstractModule):

  def _build(self, z0, is_training=False):
    self._m = snt.Linear(2, initializers={
        'w': tf.constant_initializer(1.),
        'b': lambda *unsed_args, **unused_kwargs: tf.constant([0., 1.]),
    })
    return self._m(z0)


class LossTest(tf.test.TestCase):

  def testEndToEnd(self):
    predictor = FixedNN()
    predictor = ibp.VerifiableModelWrapper(predictor)
    # Labels.
    labels = tf.constant([1], dtype=tf.int64)
    # Connect to input.
    z = tf.constant([[1, 2, 3]], dtype=tf.float32)
    predictor(z, is_training=True)
    # Input bounds.
    eps = 1.
    input_bounds = ibp.IntervalBounds(z - eps, z + eps)
    predictor.propagate_bounds(input_bounds)
    # Create output specification (that forces the first logits to be greater).
    c = tf.constant([[[1, -1]]], dtype=tf.float32)
    d = tf.constant([[0]], dtype=tf.float32)
    spec = ibp.LinearSpecification(c, d)
    # Turn elision off for more interesting results.
    spec_builder = lambda *args, **kwargs: spec(*args, collapse=False, **kwargs)  # pylint: disable=unnecessary-lambda
    # Create an attack.
    attack = ibp.UntargetedPGDAttack(
        predictor, spec, eps, num_steps=1, input_bounds=(-100., 100))
    # Build loss.
    losses = ibp.Losses(predictor, spec_builder, attack,
                        interval_bounds_loss_type='hinge',
                        interval_bounds_hinge_margin=0.)
    losses(labels)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      # We expect the worst-case logits from IBP to be [9, 4].
      # The adversarial attack should fail since logits are always [l, l + 1].
      # Similarly, the nominal predictions are correct.
      accuracy_values, loss_values = sess.run(
          [losses.scalar_metrics, losses.scalar_losses])
      self.assertAlmostEqual(1., accuracy_values.nominal_accuracy)
      self.assertAlmostEqual(0., accuracy_values.verified_accuracy)
      self.assertAlmostEqual(1., accuracy_values.attack_accuracy)
      expected_xent = 0.31326168751822947
      self.assertAlmostEqual(expected_xent, loss_values.nominal_cross_entropy)
      self.assertAlmostEqual(expected_xent, loss_values.attack_cross_entropy)
      expected_hinge = 5.
      self.assertAlmostEqual(expected_hinge, loss_values.verified_loss)


if __name__ == '__main__':
  tf.test.main()
