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

"""Tests for model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import interval_bound_propagation as ibp
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


def _build_model():
  num_classes = 3
  layer_types = (
      ('conv2d', (2, 2), 4, 'VALID', 1),
      ('activation', 'relu'),
      ('linear', 10),
      ('activation', 'relu'))
  return ibp.DNN(num_classes, layer_types)


class ModelTest(parameterized.TestCase, tf.test.TestCase):

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
    # Build another input and test reuse.
    z2 = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    z2 = tf.reshape(z, [1, 2, 2, 1])
    logits = wrapper(z2, reuse=True)
    self.assertEqual(z, wrapper.inputs)
    self.assertNotEqual(z2, wrapper.inputs)
    # Check that the verifiable modules are constructed.
    self.assertLen(wrapper.input_wrappers, 1)
    self.assertLen(wrapper.modules, 6)
    self.assertIsInstance(wrapper.modules[0].module, snt.Conv2D)
    self.assertEqual(wrapper.modules[1].module, tf.nn.relu)
    self.assertIsInstance(wrapper.modules[2].module, snt.BatchFlatten)
    self.assertIsInstance(wrapper.modules[3].module, snt.Linear)
    self.assertEqual(wrapper.modules[4].module, tf.nn.relu)
    self.assertIsInstance(wrapper.modules[5].module, snt.Linear)
    # It's a sequential network, so all nodes (including input) have fanout 1.
    self.assertEqual(wrapper.fanout_of(wrapper.input_wrappers[0]), 1)
    for module in wrapper.modules:
      self.assertEqual(wrapper.fanout_of(module), 1)
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
    self.assertLen(wrapper.input_wrappers, 1)
    self.assertLen(wrapper.modules, 5)
    # Check input has fanout 2, as it is the start of the resnet block.
    self.assertEqual(wrapper.fanout_of(wrapper.input_wrappers[0]), 2)
    for module in wrapper.modules:
      self.assertEqual(wrapper.fanout_of(module), 1)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testVerifiableModelWrapperPool(self):
    def _build(z0):
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
    def _build(z0):
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
    def _build(z0):
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

  @parameterized.named_parameters(
      ('Add', lambda z: z + z, 3),
      ('Sub', lambda z: z - z, 3),
      ('Identity', tf.identity, 3),
      ('Mul', lambda z: z * z, 3),
      ('Slice', lambda z: tf.slice(z, [0, 0], [-1, 5]), 3),
      ('StridedSlice', lambda z: z[:, :5], 3),
      ('Reshape', lambda z: tf.reshape(z, [2, 5]), 3),
      ('Const', lambda z: z + tf.ones_like(z), 5))
  def testVerifiableModelWrapperSimple(self, fn, expected_modules):
    def _build(z0):
      z = snt.Linear(10)(z0)
      z = fn(z)
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, expected_modules)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testPointlessReshape(self):
    def _build(z0):
      z = snt.Linear(10)(z0)
      z = snt.BatchFlatten()(z)  # This is a no-op; no graph nodes created.
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    # Expect the batch flatten to have been skipped.
    self.assertLen(wrapper.modules, 2)
    self.assertIsInstance(wrapper.modules[0], ibp.LinearFCWrapper)
    self.assertIsInstance(wrapper.modules[1], ibp.LinearFCWrapper)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testLeakyRelu(self):
    def _build(z0):
      z = snt.Linear(10)(z0)
      z = tf.nn.leaky_relu(z0, alpha=0.375)
      return snt.Linear(2)(z)

    z = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z)
    self.assertLen(wrapper.modules, 3)
    self.assertEqual(wrapper.modules[1].module.__name__, 'leaky_relu')
    self.assertEqual(wrapper.modules[1].parameters['alpha'], 0.375)
    # Check propagation.
    self._propagation_test(wrapper, z, logits)

  def testMultipleInputs(self):
    # Tensor to overwrite.
    def _build(z0, z1):
      return z0 + z1

    z0 = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    z1 = tf.constant([[2, 2, 4, 4]], dtype=tf.float32)
    wrapper = ibp.VerifiableModelWrapper(_build)
    logits = wrapper(z0, z1)
    input_bounds0 = ibp.IntervalBounds(z0 - 2, z0 + 1)
    input_bounds1 = ibp.IntervalBounds(z1, z1 + 10)
    output_bounds = wrapper.propagate_bounds(input_bounds0, input_bounds1)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      o, l, u = sess.run([logits, output_bounds.lower, output_bounds.upper])
      print(o, l, u)
      self.assertAlmostEqual([[3., 4., 7., 8.]], o.tolist())
      self.assertAlmostEqual([[1., 2., 5., 6.]], l.tolist())
      self.assertAlmostEqual([[14., 15., 18., 19.]], u.tolist())


if __name__ == '__main__':
  tf.test.main()
