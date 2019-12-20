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

"""Additional Sonnet modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


# Slightly altered version of snt.BatchNorm that allows to easily grab which
# mean and variance are currently in use (whether the last _build was
# invoked with is_training=True or False).
# Modifications include:
# - Removing fused option (which we do not support).
# - Removing test_local_stats (which we do not support).
# - Providing a mean and variance property.
# - Provides scale, bias properties that return None if there are none.
class BatchNorm(snt.BatchNorm):
  """Batch normalization module, including optional affine transformation."""

  def __init__(self, axis=None, offset=True, scale=False,
               decay_rate=0.999, eps=1e-3, initializers=None,
               partitioners=None, regularizers=None,
               update_ops_collection=None, name='batch_norm'):
    """Constructs a BatchNorm module. See original code for more details."""
    super(BatchNorm, self).__init__(
        axis=axis, offset=offset, scale=scale, decay_rate=decay_rate, eps=eps,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, fused=False,
        update_ops_collection=update_ops_collection, name=name)

  def _build_statistics(self, input_batch, axis, use_batch_stats, stat_dtype):
    """Builds the statistics part of the graph when using moving variance."""
    self._mean, self._variance = super(BatchNorm, self)._build_statistics(
        input_batch, axis, use_batch_stats, stat_dtype)
    return self._mean, self._variance

  def _build(self, input_batch, is_training=True, test_local_stats=False,
             reuse=False):
    """Connects the BatchNorm module into the graph.

    Args:
      input_batch: A Tensor of arbitrary dimension. By default, the final
        dimension is not reduced over when computing the minibatch statistics.
      is_training: A boolean to indicate if the module should be connected in
        training mode, meaning the moving averages are updated. Can be a Tensor.
      test_local_stats: A boolean to indicate if the statistics should be from
        the local batch. When is_training is True, test_local_stats is not used.
      reuse: If True, the statistics computed by previous call to _build
        are used and is_training is ignored. Otherwise, behaves like a normal
        batch normalization layer.

    Returns:
      A tensor with the same shape as `input_batch`.

    Raises:
      ValueError: If `axis` is not valid for the
        input shape or has negative entries.
    """
    if reuse:
      self._ensure_is_connected()
      return tf.nn.batch_normalization(
          input_batch, self._mean, self._variance, self._beta, self._gamma,
          self._eps, name='batch_norm')
    else:
      return super(BatchNorm, self)._build(input_batch, is_training,
                                           test_local_stats=test_local_stats)

  @property
  def scale(self):
    self._ensure_is_connected()
    return tf.stop_gradient(self._gamma) if self._gamma is not None else None

  @property
  def bias(self):
    self._ensure_is_connected()
    return tf.stop_gradient(self._beta) if self._beta is not None else None

  @property
  def mean(self):
    self._ensure_is_connected()
    return tf.stop_gradient(self._mean)

  @property
  def variance(self):
    self._ensure_is_connected()
    return tf.stop_gradient(self._variance)

  @property
  def epsilon(self):
    self._ensure_is_connected()
    return self._eps


class ImageNorm(snt.AbstractModule):
  """Module that does per channel normalization."""

  def __init__(self, mean, std, name='image_norm'):
    """Constructs a module that does (x[:, :, c] - mean[c]) / std[c]."""
    super(ImageNorm, self).__init__(name=name)
    if isinstance(mean, float):
      mean = [mean]
    if isinstance(std, float):
      std = [std]
    scale = []
    for s in std:
      if s <= 0.:
        raise ValueError('Cannot use negative standard deviations.')
      scale.append(1. / s)
    with self._enter_variable_scope():
      # Using broadcasting.
      self._scale = tf.constant(scale, dtype=tf.float32)
      self._offset = tf.constant(mean, dtype=tf.float32)

  def _build(self, inputs):
    return self.apply(inputs)

  @property
  def scale(self):
    return self._scale

  @property
  def offset(self):
    return self._offset

  # Provide a function that allows to use the IncreasingMonotonicWrapper.
  def apply(self, inputs):
    return (inputs - self._offset) * self._scale
