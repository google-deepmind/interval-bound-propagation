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

"""Tests for CROWN bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import interval_bound_propagation as ibp
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


def _generate_identity_spec(modules, shape, dimension=1):
  spec = ibp.LinearSpecification(tf.reshape(tf.eye(dimension), shape),
                                 prune_irrelevant=False)
  initial_bound = ibp.crown.create_initial_backward_bounds(spec, modules)
  return initial_bound


class CROWNBoundsTest(tf.test.TestCase):

  def testFCBackwardBounds(self):
    m = snt.Linear(1, initializers={
        'w': tf.constant_initializer(1.),
        'b': tf.constant_initializer(2.),
    })
    z = tf.constant([[1, 2, 3]], dtype=tf.float32)
    m(z)  # Connect to create weights.
    m = ibp.LinearFCWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    m.propagate_bounds(input_bounds)  # Create IBP bounds.
    crown_init_bounds = _generate_identity_spec([m], shape=(1, 1, 1))
    output_bounds = m.propagate_bounds(crown_init_bounds)
    concrete_bounds = output_bounds.concretize()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      lw, uw, lb, ub, cl, cu = sess.run([output_bounds.lower.w,
                                         output_bounds.upper.w,
                                         output_bounds.lower.b,
                                         output_bounds.upper.b,
                                         concrete_bounds.lower,
                                         concrete_bounds.upper])
      self.assertTrue(np.all(lw == 1.))
      self.assertTrue(np.all(lb == 2.))
      self.assertTrue(np.all(uw == 1.))
      self.assertTrue(np.all(ub == 2.))
      cl = cl.item()
      cu = cu.item()
      self.assertAlmostEqual(5., cl)
      self.assertAlmostEqual(11., cu)

  def testConv2dBackwardBounds(self):
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
    m.propagate_bounds(input_bounds)   # Create IBP bounds.
    crown_init_bounds = _generate_identity_spec([m], shape=(1, 1, 1, 1, 1))
    output_bounds = m.propagate_bounds(crown_init_bounds)
    concrete_bounds = output_bounds.concretize()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      l, u = sess.run([concrete_bounds.lower, concrete_bounds.upper])
      l = l.item()
      u = u.item()
      self.assertAlmostEqual(8., l)
      self.assertAlmostEqual(16., u)

  def testReluBackwardBounds(self):
    m = tf.nn.relu
    z = tf.constant([[-2, 3]], dtype=tf.float32)
    m = ibp.IncreasingMonotonicWrapper(m)
    input_bounds = ibp.IntervalBounds(z - 1., z + 1.)
    m.propagate_bounds(input_bounds)  # Create IBP bounds.
    crown_init_bounds = _generate_identity_spec([m], shape=(1, 2, 2),
                                                dimension=2)
    output_bounds = m.propagate_bounds(crown_init_bounds)
    concrete_bounds = output_bounds.concretize()
    with self.test_session() as sess:
      l, u = sess.run([concrete_bounds.lower, concrete_bounds.upper])
      self.assertAlmostEqual([[0., 2.]], l.tolist())
      self.assertAlmostEqual([[0., 4.]], u.tolist())

if __name__ == '__main__':
  tf.test.main()
