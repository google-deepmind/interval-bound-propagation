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

"""Tests for bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


class BoundsTest(tf.test.TestCase):

  def testFCIntervalBounds(self):
    m = snt.Linear(1, initializers={
        'w': tf.constant_initializer(1.),
        'b': tf.constant_initializer(2.),
    })
    z = tf.constant([[1, 2, 3]], dtype=tf.float32)
    m(z)  # Connect to create weights.
    m = ibp.LinearFCWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    output_bounds = input_bounds.propagate_through(m)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      l = l.item()
      u = u.item()
      self.assertAlmostEqual(5., l)
      self.assertAlmostEqual(11., u)

  def testConv2dIntervalBounds(self):
    m = snt.Conv2D(
        output_channels=1,
        kernel_shape=(2, 2),
        padding='VALID',
        stride=1,
        use_bias=True,
        initializers={
            'w': tf.constant_initializer(1.),
            'b': tf.constant_initializer(2.),
        })
    z = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 2, 1])
    m(z)  # Connect to create weights.
    m = ibp.LinearConv2dWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    output_bounds = input_bounds.propagate_through(m)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      l = l.item()
      u = u.item()
      self.assertAlmostEqual(8., l)
      self.assertAlmostEqual(16., u)

  def testReluIntervalBounds(self):
    m = tf.nn.relu
    z = tf.constant([[-2, 3]], dtype=tf.float32)
    m = ibp.MonotonicWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    output_bounds = input_bounds.propagate_through(m)
    with self.test_session() as sess:
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      self.assertAlmostEqual([[0., 2.]], l.tolist())
      self.assertAlmostEqual([[0., 4.]], u.tolist())

  def testBatchNormIntervalBounds(self):
    z = tf.constant([[1, 2, 3]], dtype=tf.float32)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    g = tf.reshape(tf.range(-1, 2, dtype=tf.float32), [1, 3])
    b = tf.reshape(tf.range(3, dtype=tf.float32), [1, 3])
    batch_norm = ibp.BatchNorm(scale=True, offset=True, eps=0., initializers={
        'gamma': lambda *args, **kwargs: g,
        'beta': lambda *args, **kwargs: b,
        'moving_mean': tf.constant_initializer(1.),
        'moving_variance': tf.constant_initializer(4.),
    })
    batch_norm(z, is_training=False)
    batch_norm = ibp.BatchNormWrapper(batch_norm)
    # Test propagation.
    output_bounds = input_bounds.propagate_through(batch_norm)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      self.assertAlmostEqual([[-.5, 1., 2.5]], l.tolist())
      self.assertAlmostEqual([[.5, 1., 3.5]], u.tolist())


if __name__ == '__main__':
  tf.test.main()
