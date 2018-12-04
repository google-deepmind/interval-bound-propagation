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

"""Defines the output specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import sonnet as snt
import tensorflow as tf


class LinearSpecification(snt.AbstractModule):
  """Linear specifications: c^T * z_K + d <= 0."""

  def __init__(self, c, d=None):
    """Builds a linear specification module."""
    super(LinearSpecification, self).__init__(name='specs')
    # c has shape [batch_size, num_specifications, num_outputs]
    # d has shape [batch_size, num_specifications]
    # Some specifications may be irrelevant (not a function of the output).
    # We automatically remove them for clarity. We expect the number of
    # irrelevant specs to be equal for all elements of a batch.
    # Shape is [batch_size, num_specifications]
    irrelevant = tf.equal(tf.reduce_sum(
        tf.cast(tf.abs(c) > 1e-6, tf.int32), axis=-1, keepdims=True), 0)
    batch_size = tf.shape(c)[0]
    num_outputs = tf.shape(c)[2]
    irrelevant = tf.tile(irrelevant, [1, 1, num_outputs])
    self._c = tf.reshape(
        tf.boolean_mask(c, tf.logical_not(irrelevant)),
        [batch_size, -1, num_outputs])
    self._d = d

  def _build(self, modules, collapse=True):
    """Outputs specification value."""
    # inputs have shape [batch_size, num_outputs].
    if not collapse:
      logging.info('Elision of last layer disabled.')
      bounds = modules[-1].output_bounds
      w = self._c
      b = self._d
    else:
      logging.info('Elision of last layer active.')
      # Collapse the last layer.
      bounds = modules[-1].input_bounds
      w = modules[-1].module.w
      b = modules[-1].module.b
      w = tf.einsum('ijk,lk->ijl', self._c, w)
      b = tf.einsum('ijk,k->ij', self._c, b)
      if self._d is not None:
        b += self._d

    # Maximize z * w + b s.t. lower <= z <= upper.
    c = (bounds.lower + bounds.upper) / 2.
    r = (bounds.upper - bounds.lower) / 2.
    c = tf.einsum('ij,ikj->ik', c, w)
    if b is not None:
      c += b
    r = tf.einsum('ij,ikj->ik', r, tf.abs(w))
    self._output = c + r

    # output has shape [batch_size, num_specifications].
    return self._output

  @property
  def c(self):
    return self._c

  @property
  def d(self):
    return self._d
