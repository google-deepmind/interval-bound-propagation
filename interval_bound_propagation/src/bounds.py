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

"""Definition of input bounds to each layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from interval_bound_propagation.src import verifiable_wrapper
import sonnet as snt
import tensorflow as tf


class AbstractBounds(object):
  """Abstract bounds class."""

  __metaclass__ = abc.ABCMeta

  def propagate_through(self, wrapper, *args):
    """Propagates bounds through a verifiable wrapper.

    Args:
      wrapper: `verifiable_wrapper.VerifiableWrapper`
      *args: Additional arguments passed down to downstream callbacks.

    Returns:
      New bounds.
    """
    module = wrapper.module
    if isinstance(wrapper, verifiable_wrapper.LinearFCWrapper):
      w = module.w
      b = module.b if module.has_bias else None
      return self._linear(w, b)
    elif isinstance(wrapper, verifiable_wrapper.LinearConv2dWrapper):
      w = module.w
      b = module.b if module.has_bias else None
      padding = module.padding
      strides = module.stride[1:-1]
      return self._conv2d(w, b, padding, strides)
    elif isinstance(wrapper, verifiable_wrapper.MonotonicWrapper):
      return self._monotonic_fn(module, *args)
    elif isinstance(wrapper, verifiable_wrapper.BatchNormWrapper):
      return self._batch_norm(module.mean, module.variance, module.scale,
                              module.bias, module.epsilon)
    elif isinstance(wrapper, verifiable_wrapper.BatchFlattenWrapper):
      return self._batch_flatten()
    else:
      raise NotImplementedError('{} not supported.'.format(
          wrapper.__class__.__name__))

  def _raise_not_implemented(self, name):
    raise NotImplementedError(
        '{} modules are not supported by "{}".'.format(
            name, self.__class__.__name__))

  def _linear(self, w, b):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Linear')

  def _conv2d(self, w, b, padding, strides):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Conv2D')

  def _monotonic_fn(self, fn, *args):  # pylint: disable=unused-argument
    self._raise_not_implemented(fn.__name__)

  def _batch_norm(self, mean, variance, scale, bias, epsilon):  # pylint: disable=unused-argument
    self._raise_not_implemented('ibp.BatchNorm')

  def _batch_flatten(self):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.BatchFlatten')


class IntervalBounds(AbstractBounds):
  """Axis-aligned bounding box."""

  def __init__(self, lower, upper):
    self._lower = lower
    self._upper = upper

  @property
  def lower(self):
    return self._lower

  @property
  def upper(self):
    return self._upper

  def _linear(self, w, b):
    c = (self.lower + self.upper) / 2.
    r = (self.upper - self.lower) / 2.
    c = tf.matmul(c, w)
    if b is not None:
      c = c + b
    r = tf.matmul(r, tf.abs(w))
    return IntervalBounds(c - r, c + r)

  def _conv2d(self, w, b, padding, strides):
    c = (self.lower + self.upper) / 2.
    r = (self.upper - self.lower) / 2.
    c = tf.nn.convolution(c, w, padding=padding, strides=strides)
    if b is not None:
      c = c + b
    r = tf.nn.convolution(r, tf.abs(w), padding=padding, strides=strides)
    return IntervalBounds(c - r, c + r)

  def _monotonic_fn(self, fn, *args):
    args_lower = [self.lower] + [a.lower for a in args]
    args_upper = [self.upper] + [a.upper for a in args]
    return IntervalBounds(fn(*args_lower), fn(*args_upper))

  def _batch_norm(self, mean, variance, scale, bias, epsilon):
    # Element-wise multiplier.
    multiplier = tf.rsqrt(variance + epsilon)
    if scale is not None:
      multiplier *= scale
    w = multiplier
    # Element-wise bias.
    b = -multiplier * mean
    if bias is not None:
      b += bias
    b = tf.squeeze(b, axis=0)
    # Because the scale might be negative, we need to apply a strategy similar
    # to linear.
    c = (self.lower + self.upper) / 2.
    r = (self.upper - self.lower) / 2.
    c = tf.multiply(c, w) + b
    r = tf.multiply(r, tf.abs(w))
    return IntervalBounds(c - r, c + r)

  def _batch_flatten(self):
    return IntervalBounds(snt.BatchFlatten()(self.lower),
                          snt.BatchFlatten()(self.upper))
