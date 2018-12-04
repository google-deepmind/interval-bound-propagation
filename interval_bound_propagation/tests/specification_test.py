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

"""Tests for specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import interval_bound_propagation as ibp
import numpy as np
import tensorflow as tf


MockLinearModule = collections.namedtuple('MockLinearModule', ['w', 'b'])
MockModule = collections.namedtuple(
    'MockModule', ['input_bounds', 'output_bounds', 'module'])


def _build_spec_input():
  # Specifications expects a list of objects with output_bounds or input_bounds
  # attributes.
  w = np.identity(2, dtype=np.float32)
  b = np.ones(2, dtype=np.float32)
  snt_module = MockLinearModule(tf.constant(w), tf.constant(b))
  z_lower = np.array([[1, 2]], dtype=np.float32)
  z_upper = np.array([[3, 4]], dtype=np.float32)
  input_bounds = ibp.IntervalBounds(tf.constant(z_lower), tf.constant(z_upper))
  z_lower += b
  z_upper += b
  output_bounds = ibp.IntervalBounds(tf.constant(z_lower), tf.constant(z_upper))
  return [MockModule(input_bounds, output_bounds, snt_module)]


class SpecificationTest(tf.test.TestCase):

  def testLinearSpecification(self):
    # c has shape [batch_size, num_specifications, num_outputs]
    # d has shape [batch_size, num_specifications]
    c = tf.constant([[[1, 2]]], dtype=tf.float32)
    d = tf.constant([[3]], dtype=tf.float32)
    # The above is equivalent to z_{K,1} + 2 * z_{K,2} + 3 <= 0
    spec = ibp.LinearSpecification(c, d)
    modules = _build_spec_input()
    values = spec(modules, collapse=False)
    values_collapse = spec(modules, collapse=True)
    with self.test_session() as sess:
      self.assertAlmostEqual(17., sess.run(values).item())
      self.assertAlmostEqual(17., sess.run(values_collapse).item())


if __name__ == '__main__':
  tf.test.main()
