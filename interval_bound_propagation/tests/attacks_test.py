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

"""Tests for attacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow.compat.v1 as tf


class MockWithIsTraining(object):
  """Mock wrapper around the predictor network."""

  def __init__(self, module, test):
    self._module = module
    self._test = test

  def __call__(self, z0, is_training=False):
    # is_training should be False.
    self._test.assertFalse(is_training)
    return self._module(z0)


class MockWithoutIsTraining(object):
  """Mock wrapper around the predictor network."""

  def __init__(self, module, test):
    self._module = module
    self._test = test

  def __call__(self, z0):
    return self._module(z0)


class AttacksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('UntargetedWithGradientDescent', MockWithIsTraining,
       ibp.UntargetedPGDAttack, ibp.UnrolledGradientDescent, 1.),
      ('UntargetedWithAdam', MockWithIsTraining,
       ibp.UntargetedPGDAttack, ibp.UnrolledAdam, 1.),
      ('MultiTargetedWithGradientDescent', MockWithIsTraining,
       ibp.MultiTargetedPGDAttack, ibp.UnrolledGradientDescent, 1.),
      ('MultiTargetedWithAdam', MockWithIsTraining,
       ibp.MultiTargetedPGDAttack, ibp.UnrolledAdam, 1.),
      ('DiverseEpsilon', MockWithIsTraining,
       ibp.MultiTargetedPGDAttack, ibp.UnrolledAdam, [1., 1.]),
      ('WithoutIsTraining', MockWithoutIsTraining,
       ibp.UntargetedPGDAttack, ibp.UnrolledGradientDescent, 1.),
      ('Restarted', MockWithIsTraining,
       ibp.UntargetedPGDAttack, ibp.UnrolledGradientDescent, 1., True),
      ('SPSA', MockWithIsTraining,
       ibp.UntargetedPGDAttack, ibp.UnrolledSPSAAdam, 1.))
  def testEndToEnd(self, predictor_cls, attack_cls, optimizer_cls, epsilon,
                   restarted=False):
    # l-\infty norm of perturbation ball.
    if isinstance(epsilon, list):
      # We test the ability to have different epsilons across dimensions.
      epsilon = tf.constant([epsilon], dtype=tf.float32)
    bounds = (-.5, 2.5)
    # Create a simple network.
    m = snt.Linear(1, initializers={
        'w': tf.constant_initializer(1.),
        'b': tf.constant_initializer(1.),
    })
    z = tf.constant([[1, 2]], dtype=tf.float32)
    predictor = predictor_cls(m, self)
    # Not important for the test but needed.
    labels = tf.constant([1], dtype=tf.int64)

    # We create two attacks to maximize and then minimize the output.
    max_spec = ibp.LinearSpecification(tf.constant([[[1.]]]))
    max_attack = attack_cls(predictor, max_spec, epsilon, input_bounds=bounds,
                            optimizer_builder=optimizer_cls)
    if restarted:
      max_attack = ibp.RestartedAttack(max_attack, num_restarts=10)
    z_max = max_attack(z, labels)
    min_spec = ibp.LinearSpecification(tf.constant([[[-1.]]]))
    min_attack = attack_cls(predictor, min_spec, epsilon, input_bounds=bounds,
                            optimizer_builder=optimizer_cls)
    if restarted:
      min_attack = ibp.RestartedAttack(min_attack, num_restarts=10)
    z_min = min_attack(z, labels)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      z_max_values, z_min_values = sess.run([z_max, z_min])
      z_max_values = z_max_values[0]
      z_min_values = z_min_values[0]
      self.assertAlmostEqual(2., z_max_values[0])
      self.assertAlmostEqual(2.5, z_max_values[1])
      self.assertAlmostEqual(0., z_min_values[0])
      self.assertAlmostEqual(1., z_min_values[1])


if __name__ == '__main__':
  tf.test.main()
