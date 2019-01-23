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


def _build_classification_specification(label, num_classes):
  """Returns a LinearSpecification for adversarial classification."""
  # Pre-construct the specifications of the different classes.
  eye = np.eye(num_classes - 1)
  specifications = []
  for i in range(num_classes):
    specifications.append(np.concatenate(
        [eye[:, :i], -np.ones((num_classes - 1, 1)), eye[:, i:]], axis=1))
  specifications = np.array(specifications, dtype=np.float32)
  specifications = tf.constant(specifications)
  # We can then use gather.
  c = tf.gather(specifications, label)
  # By construction all specifications are relevant.
  d = tf.zeros(shape=(tf.shape(label)[0], num_classes - 1))
  return ibp.LinearSpecification(c, d, prune_irrelevant=False)


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

  def testEquivalenceLinearClassification(self):
    num_classes = 3
    def _build_model():
      layer_types = (
          ('conv2d', (2, 2), 4, 'VALID', 1),
          ('activation', 'relu'),
          ('linear', 10),
          ('activation', 'relu'))
      return ibp.DNN(num_classes, layer_types)

    # Input.
    batch_size = 100
    width = height = 2
    channels = 3
    num_restarts = 10
    z = tf.random.uniform((batch_size, height, width, channels),
                          minval=-1., maxval=1., dtype=tf.float32)
    y = tf.random.uniform((batch_size,), minval=0, maxval=num_classes,
                          dtype=tf.int64)
    predictor = _build_model()
    predictor = ibp.VerifiableModelWrapper(predictor)
    logits = predictor(z)
    random_logits1 = tf.random.uniform((num_restarts, batch_size, num_classes))
    random_logits2 = tf.random.uniform((num_restarts, num_classes - 1,
                                        batch_size, num_classes))
    input_bounds = ibp.IntervalBounds(z - 2., z + 4.)
    predictor.propagate_bounds(input_bounds)

    # Specifications.
    s1 = ibp.ClassificationSpecification(y, num_classes)
    s2 = _build_classification_specification(y, num_classes)
    def _build_values(s):
      return [
          s(predictor.modules, collapse=False),
          s(predictor.modules, collapse=True),
          s.evaluate(logits),
          s.evaluate(random_logits1),
          s.evaluate(random_logits2)
      ]
    v1 = _build_values(s1)
    v2 = _build_values(s2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output1, output2 = sess.run([v1, v2])
    for a, b in zip(output1, output2):
      self.assertTrue(np.all(np.abs(a - b) < 1e-5))


if __name__ == '__main__':
  tf.test.main()
