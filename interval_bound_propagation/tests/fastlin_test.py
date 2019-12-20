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

"""Tests for symbolic bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import interval_bound_propagation as ibp
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class SymbolicBoundsTest(parameterized.TestCase, tf.test.TestCase):

  def testConvertSymbolicBounds(self):
    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 2])
    b = ibp.SymbolicBounds.convert(z)
    for l in (b.lower, b.upper):
      self.assertEqual([1, 4, 2, 2], l.w.shape.as_list())
      self.assertEqual([1, 2, 2], l.b.shape.as_list())
      self.assertEqual([1, 4], l.lower.shape.as_list())
      self.assertEqual([1, 4], l.upper.shape.as_list())

  def testFCSymbolicBounds(self):
    m = snt.Linear(1, initializers={
        'w': tf.constant_initializer(1.),
        'b': tf.constant_initializer(2.),
    })
    z = tf.constant([[1, 2, 3]], dtype=tf.float32)
    m(z)  # Connect to create weights.
    m = ibp.LinearFCWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    input_bounds = ibp.SymbolicBounds.convert(input_bounds)
    output_bounds = m.propagate_bounds(input_bounds)
    concrete_bounds = ibp.IntervalBounds.convert(output_bounds)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u, cl, cu = sess.run([output_bounds.lower, output_bounds.upper,
                               concrete_bounds.lower, concrete_bounds.upper])
      self.assertTrue(np.all(l.w == 1.))
      self.assertTrue(np.all(l.b == 2.))
      self.assertAlmostEqual([[0, 1, 2]], l.lower.tolist())
      self.assertAlmostEqual([[2, 3, 4]], l.upper.tolist())
      self.assertTrue(np.all(u.w == 1.))
      self.assertTrue(np.all(u.b == 2.))
      self.assertAlmostEqual([[0, 1, 2]], u.lower.tolist())
      self.assertAlmostEqual([[2, 3, 4]], u.upper.tolist())
      cl = cl.item()
      cu = cu.item()
      self.assertAlmostEqual(5., cl)
      self.assertAlmostEqual(11., cu)

  def testConv2dSymbolicBounds(self):
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
    input_bounds = ibp.SymbolicBounds.convert(input_bounds)
    output_bounds = m.propagate_bounds(input_bounds)
    output_bounds = ibp.IntervalBounds.convert(output_bounds)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      l = l.item()
      u = u.item()
      self.assertAlmostEqual(8., l)
      self.assertAlmostEqual(16., u)

  def testConv1dSymbolicBounds(self):
    m = snt.Conv1D(
        output_channels=1,
        kernel_shape=(2),
        padding='VALID',
        stride=1,
        use_bias=True,
        initializers={
            'w': tf.constant_initializer(1.),
            'b': tf.constant_initializer(3.),
        })
    z = tf.constant([3, 4], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 1])
    m(z)  # Connect to create weights.
    m = ibp.LinearConv1dWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    input_bounds = ibp.SymbolicBounds.convert(input_bounds)
    output_bounds = m.propagate_bounds(input_bounds)
    output_bounds = ibp.IntervalBounds.convert(output_bounds)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      l = l.item()
      u = u.item()
      self.assertAlmostEqual(8., l)
      self.assertAlmostEqual(12., u)

  def testReluSymbolicBounds(self):
    m = tf.nn.relu
    z = tf.constant([[-2, 3]], dtype=tf.float32)
    m = ibp.IncreasingMonotonicWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    input_bounds = ibp.SymbolicBounds.convert(input_bounds)
    output_bounds = m.propagate_bounds(input_bounds)
    output_bounds = ibp.IntervalBounds.convert(output_bounds)
    with self.test_session() as sess:
      l, u = sess.run([output_bounds.lower, output_bounds.upper])
      self.assertAlmostEqual([[0., 2.]], l.tolist())
      self.assertAlmostEqual([[0., 4.]], u.tolist())


if __name__ == '__main__':
  tf.test.main()
