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

"""Definition of input bounds to each layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools

import six
import sonnet as snt
import tensorflow.compat.v1 as tf


@six.add_metaclass(abc.ABCMeta)
class AbstractBounds(object):
  """Abstract bounds class."""

  def __init__(self):
    self._update_cache_op = None

  @classmethod
  @abc.abstractmethod
  def convert(cls, bounds):
    """Converts another bound type to this type."""

  @abc.abstractproperty
  def shape(self):
    """Returns shape (as list) of the tensor, including batch dimension."""

  def concretize(self):
    return self

  def _raise_not_implemented(self, name):
    raise NotImplementedError(
        '{} modules are not supported by "{}".'.format(
            name, self.__class__.__name__))

  def apply_linear(self, wrapper, w, b):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Linear')

  def apply_conv1d(self, wrapper, w, b, padding, stride):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Conv1D')

  def apply_conv2d(self, wrapper, w, b, padding, strides):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Conv2D')

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):  # pylint: disable=unused-argument
    self._raise_not_implemented(fn.__name__)

  def apply_piecewise_monotonic_fn(self, wrapper, fn, boundaries, *args):  # pylint: disable=unused-argument
    self._raise_not_implemented(fn.__name__)

  def apply_batch_norm(self, wrapper, mean, variance, scale, bias, epsilon):  # pylint: disable=unused-argument
    self._raise_not_implemented('ibp.BatchNorm')

  def apply_batch_reshape(self, wrapper, shape):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.BatchReshape')

  def apply_softmax(self, wrapper):  # pylint: disable=unused-argument
    self._raise_not_implemented('tf.nn.softmax')

  @property
  def update_cache_op(self):
    """TF op to update cached bounds for re-use across session.run calls."""
    if self._update_cache_op is None:
      raise ValueError('Bounds not cached: enable_caching() not called.')
    return self._update_cache_op

  def enable_caching(self):
    """Enables caching the bounds for re-use across session.run calls."""
    if self._update_cache_op is not None:
      raise ValueError('Bounds already cached: enable_caching() called twice.')
    self._update_cache_op = self._set_up_cache()

  def _set_up_cache(self):
    """Replace fields with cached versions.

    Returns:
      TensorFlow op to update the cache.
    """
    return tf.no_op()  # By default, don't cache.

  def _cache_with_update_op(self, tensor):
    """Creates non-trainable variable to cache the tensor across sess.run calls.

    Args:
      tensor: Tensor to cache.

    Returns:
      cached_tensor: Non-trainable variable to contain the cached value
        of `tensor`.
      update_op: TensorFlow op to re-evaluate `tensor` and assign the result
        to `cached_tensor`.
    """
    cached_tensor = tf.get_variable(
        tensor.name.replace(':', '__') + '_ibp_cache',
        shape=tensor.shape, dtype=tensor.dtype, trainable=False)
    update_op = tf.assign(cached_tensor, tensor)
    return cached_tensor, update_op


class IntervalBounds(AbstractBounds):
  """Axis-aligned bounding box."""

  def __init__(self, lower, upper):
    super(IntervalBounds, self).__init__()
    self._lower = lower
    self._upper = upper

  @property
  def lower(self):
    return self._lower

  @property
  def upper(self):
    return self._upper

  @property
  def shape(self):
    return self.lower.shape.as_list()

  def __iter__(self):
    yield self.lower
    yield self.upper

  @classmethod
  def convert(cls, bounds):
    if isinstance(bounds, tf.Tensor):
      return cls(bounds, bounds)
    bounds = bounds.concretize()
    if not isinstance(bounds, cls):
      raise ValueError('Cannot convert "{}" to "{}"'.format(bounds,
                                                            cls.__name__))
    return bounds

  def apply_linear(self, wrapper, w, b):
    return self._affine(w, b, tf.matmul)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    return self._affine(w, b, tf.nn.conv1d, padding=padding, stride=stride)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    return self._affine(w, b, tf.nn.convolution,
                        padding=padding, strides=strides)

  def _affine(self, w, b, fn, **kwargs):
    c = (self.lower + self.upper) / 2.
    r = (self.upper - self.lower) / 2.
    c = fn(c, w, **kwargs)
    if b is not None:
      c = c + b
    r = fn(r, tf.abs(w), **kwargs)
    return IntervalBounds(c - r, c + r)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    args_lower = [self.lower] + [a.lower for a in args]
    args_upper = [self.upper] + [a.upper for a in args]
    return IntervalBounds(fn(*args_lower), fn(*args_upper))

  def apply_piecewise_monotonic_fn(self, wrapper, fn, boundaries, *args):
    valid_values = []
    for a in [self] + list(args):
      vs = []
      vs.append(a.lower)
      vs.append(a.upper)
      for b in boundaries:
        vs.append(
            tf.maximum(a.lower, tf.minimum(a.upper, b * tf.ones_like(a.lower))))
      valid_values.append(vs)
    outputs = []
    for inputs in itertools.product(*valid_values):
      outputs.append(fn(*inputs))
    outputs = tf.stack(outputs, axis=-1)
    return IntervalBounds(tf.reduce_min(outputs, axis=-1),
                          tf.reduce_max(outputs, axis=-1))

  def apply_batch_norm(self, wrapper, mean, variance, scale, bias, epsilon):
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

  def apply_batch_reshape(self, wrapper, shape):
    return IntervalBounds(snt.BatchReshape(shape)(self.lower),
                          snt.BatchReshape(shape)(self.upper))

  def apply_softmax(self, wrapper):
    ub = self.upper
    lb = self.lower
    # Keep diagonal and take opposite bound for non-diagonals.
    lbs = tf.matrix_diag(lb) + tf.expand_dims(ub, axis=-2) - tf.matrix_diag(ub)
    ubs = tf.matrix_diag(ub) + tf.expand_dims(lb, axis=-2) - tf.matrix_diag(lb)
    # Get diagonal entries after softmax operation.
    ubs = tf.matrix_diag_part(tf.nn.softmax(ubs))
    lbs = tf.matrix_diag_part(tf.nn.softmax(lbs))
    return IntervalBounds(lbs, ubs)

  def _set_up_cache(self):
    self._lower, update_lower_op = self._cache_with_update_op(self._lower)
    self._upper, update_upper_op = self._cache_with_update_op(self._upper)
    return tf.group([update_lower_op, update_upper_op])


