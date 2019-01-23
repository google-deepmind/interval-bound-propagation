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

"""Tests for model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import interval_bound_propagation as ibp
import numpy as np
import sonnet as snt
import tensorflow as tf


def _build_model():
  num_classes = 3
  layer_types = (
      ('conv2d', (2, 2), 4, 'VALID', 1),
      ('activation', 'relu'),
      ('linear', 10),
      ('activation', 'relu'))
  return ibp.DNN(num_classes, layer_types)


class ModelTest(tf.test.TestCase):

  def assertLen(self, container, expected_len):
    self.assertEqual(expected_len, len(container))

  def testDNN(self):
    predictor = _build_model()
    # Input.
    z = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 2, 1])
    predictor(z)
    # Verify the variables that are created.
    expected_shapes = {
        'predictor/conv2d_0/w:0': (2, 2, 1, 4),
        'predictor/conv2d_0/b:0': (4,),
        'predictor/linear_0/w:0': (4, 10),
        'predictor/linear_0/b:0': (10,),
        'predictor/linear_1/w:0': (10, 3),
        'predictor/linear_1/b:0': (3,),
    }
    for v in predictor.get_variables():
      self.assertEqual(expected_shapes[v.name], v.shape)

  def _propagation_test(self, wrapper, inputs, outputs):
    input_bounds = ibp.IntervalBounds(inputs, inputs)
    output_bounds = wrapper.propagate_bounds(input_bounds)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      o, l, u = sess.run([outputs, output_bounds.lower, output_bounds.upper])
      self.assertAlmostEqual(o.tolist(), l.tolist())
      self.assertAlmostEqual(o.tolist(), u.tolist())

  def testVerifiableModelWrapperDNN(self):
    predictor = _build_model()
    # Input.
    z = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 2, 1])
    wrapper = ibp.VerifiableModelWrapper(predictor)
    wrapper(z)
    # Verify basic wrapping.
    self.assertEqual(predictor, wrapper.wrapped_network)
    self.assertEqual(3, wrapper.output_size)
    self.assertEqual((1, 3), tuple(wrapper.logits.shape.as_list()))
    self.assertEqual(z, wrapper.inputs)
    # Build another input and test passthrough.
    z2 = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    z2 = tf.reshape(z, [1, 2, 2, 1])
    logits = wrapper(z2, passthrough=True)
    self.assertEqual(z, wrapper.inputs)
    self.assertNotEqual(z2, wrapper.inputs)
    # Check that the verifiable modules are constructed.
    self.assertLen(wrapper.modules, 6)
    self.assertTrue(isinstance(wrapper.modules[0].module, snt.Conv2D))
    self.assertEqual(wrapper.modules[1].module, tf.nn.relu)
    self.assertTrue(isinstance(wrapper.modules[2].module, snt.BatchFlatten))
    self.assertTrue(isinstance(wrapper.modules[3].module, snt.Linear))
    self.assertEqual(wrapper.modules[4].module, tf.nn.relu)
    self.assertTrue(isinstance(wrapper.modules[5].module, snt.Linear))
    # Check propagation.
    self._propagation_test(wrapper, z2, logits)

  def testVerifiableModelWrapperResnet(self):
    def _build(z0, is_training=False):  # pylint: disable=unused-argument
      input_size = np.prod(z0.shape[1:])
      # We make a resnet-like structure.
      z = snt.Linear(input_size)(z0)
      z_left = tf.nn.relu(z)
      z_left = snt.Linear(input_size)(z_left)
      z = z_left + z0
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, 5)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testVerifiableModelWrapperPool(self):
    def _build(z0, is_training=False):  # pylint: disable=unused-argument
      z = tf.reduce_mean(z0, axis=1, keep_dims=True)
      z = tf.reduce_max(z, axis=2, keep_dims=False)
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    z = tf.reshape(z, [1, 2, 2])
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, 3)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testVerifiableModelWrapperConcat(self):
    def _build(z0, is_training=False):  # pylint: disable=unused-argument
      z = snt.Linear(10)(z0)
      z = tf.concat([z, z0], axis=1)
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, 3)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testVerifiableModelWrapperExpandAndSqueeze(self):
    def _build(z0, is_training=False):  # pylint: disable=unused-argument
      z = snt.Linear(10)(z0)
      z = tf.expand_dims(z, axis=-1)
      z = tf.squeeze(z, axis=-1)
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, 4)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)


if __name__ == '__main__':
  tf.test.main()
