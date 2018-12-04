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

"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import interval_bound_propagation as ibp
import numpy as np
import tensorflow as tf


def _get_inputs(dtype=tf.float32):
  v = np.array(range(6), dtype=dtype.as_numpy_dtype)
  input_v = np.array([v] * 7)
  inputs = tf.constant(input_v)
  return v, input_v, inputs


class LayersTest(tf.test.TestCase):

  def assertBetween(self, value, minv, maxv):
    """Asserts that value is between minv and maxv (inclusive)."""
    self.assertTrue(minv <= value)
    self.assertTrue(maxv >= value)

  # Subset of the tests in sonnet/python/modules/batch_norm_test.py.
  def testBatchNormUpdateImproveStatistics(self):
    """Test that updating the moving_mean improves statistics."""
    _, _, inputs = _get_inputs()
    # Use small decay_rate to update faster.
    bn = ibp.BatchNorm(offset=False, scale=False, decay_rate=0.1)
    out1 = bn(inputs, is_training=False)
    # Build the update ops.
    bn(inputs, is_training=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_v = sess.run(out1)
      # Before updating the moving_mean the results are off.
      self.assertBetween(np.max(np.abs(np.zeros([7, 6]) - out_v)), 2, 5)
      sess.run(tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
      # After updating the moving_mean the results are better.
      out_v = sess.run(out1)
      self.assertBetween(np.max(np.abs(np.zeros([7, 6]) - out_v)), 1, 2)

  def testImageNorm(self):
    mean = [4, 0, -4]
    std = [1., 2., 4.]
    image = tf.constant(4., shape=[10, 2, 2, 3])
    normalized_image = ibp.ImageNorm(mean, std)(image)

    with self.test_session() as sess:
      out_image = sess.run(normalized_image)
      self.assertTrue(np.all(np.isclose(out_image[:, :, :, 0], 0.)))
      self.assertTrue(np.all(np.isclose(out_image[:, :, :, 1], 2.)))
      self.assertTrue(np.all(np.isclose(out_image[:, :, :, 2], 2.)))


if __name__ == '__main__':
  tf.test.main()
