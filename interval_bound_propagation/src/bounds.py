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
import collections
import itertools

from absl import logging
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
    elif isinstance(wrapper, verifiable_wrapper.IncreasingMonotonicWrapper):
      return self._increasing_monotonic_fn(module, *args)
    elif isinstance(wrapper, verifiable_wrapper.PiecewiseMonotonicWrapper):
      return self._piecewise_monotonic_fn(module, wrapper.boundaries, *args)
    elif isinstance(wrapper, verifiable_wrapper.BatchNormWrapper):
      return self._batch_norm(module.mean, module.variance, module.scale,
                              module.bias, module.epsilon)
    elif isinstance(wrapper, verifiable_wrapper.BatchFlattenWrapper):
      return self._batch_flatten()
    elif isinstance(wrapper, verifiable_wrapper.SoftmaxWrapper):
      return self._softmax()
    else:
      raise NotImplementedError('{} not supported.'.format(
          wrapper.__class__.__name__))

  @staticmethod
  @abc.abstractmethod
  def convert(bounds):
    """Converts another bound type to this type."""

  def _raise_not_implemented(self, name):
    raise NotImplementedError(
        '{} modules are not supported by "{}".'.format(
            name, self.__class__.__name__))

  def _linear(self, w, b):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Linear')

  def _conv2d(self, w, b, padding, strides):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.Conv2D')

  def _increasing_monotonic_fn(self, fn, *args):  # pylint: disable=unused-argument
    self._raise_not_implemented(fn.__name__)

  def _piecewise_monotonic_fn(self, fn, boundaries, *args):  # pylint: disable=unused-argument
    self._raise_not_implemented(fn.__name__)

  def _batch_norm(self, mean, variance, scale, bias, epsilon):  # pylint: disable=unused-argument
    self._raise_not_implemented('ibp.BatchNorm')

  def _batch_flatten(self):  # pylint: disable=unused-argument
    self._raise_not_implemented('snt.BatchFlatten')

  def _softmax(self):
    self._raise_not_implemented('tf.nn.softmax')


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

  @staticmethod
  def convert(bounds):
    if isinstance(bounds, tf.Tensor):
      return IntervalBounds(bounds, bounds)
    if isinstance(bounds, SymbolicBounds):
      return IntervalBounds(*bounds.concretize())
    if not isinstance(bounds, IntervalBounds):
      raise ValueError('Cannot convert "{}" to "IntervalBounds"'.format(bounds))
    return bounds

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

  def _increasing_monotonic_fn(self, fn, *args):
    args_lower = [self.lower] + [a.lower for a in args]
    args_upper = [self.upper] + [a.upper for a in args]
    return IntervalBounds(fn(*args_lower), fn(*args_upper))

  def _piecewise_monotonic_fn(self, fn, boundaries, *args):
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

  def _softmax(self):
    ub = self.upper
    lb = self.lower
    # Keep diagonal and take opposite bound for non-diagonals.
    lbs = tf.matrix_diag(lb) + tf.expand_dims(ub, axis=-2) - tf.matrix_diag(ub)
    ubs = tf.matrix_diag(ub) + tf.expand_dims(lb, axis=-2) - tf.matrix_diag(lb)
    # Get diagonal entries after softmax operation.
    ubs = tf.matrix_diag_part(tf.nn.softmax(ubs))
    lbs = tf.matrix_diag_part(tf.nn.softmax(lbs))
    return IntervalBounds(lbs, ubs)


# Holds the linear expressions serving as bounds.
# w: [batch_size, input_size, output_shape] storing the weights.
# b: [batch_size, output_shape] storing the bias.
# lower: [batch_size, input_size] storing the lower bounds on inputs.
# upper: [batch_size, input_size] storing the upper bounds on inputs.
# `lower` and `upper` tensors are always flattened representations of the
# original inputs.
LinearExpression = collections.namedtuple(
    'LinearExpression', ['w', 'b', 'lower', 'upper'])


class SymbolicBounds(AbstractBounds):
  """Fast-Lin bounds (https://arxiv.org/abs/1804.09699)."""

  def __init__(self, lower, upper):
    self._lower = lower
    self._upper = upper

  @property
  def lower(self):
    return self._lower

  @property
  def upper(self):
    return self._upper

  def concretize(self):
    """Returns lower and upper interval bounds."""
    shape = self.lower.b.shape
    if len(shape) == 2:
      return self._concretize_bounds_1d(self.lower, self.upper)
    elif len(shape) == 4:
      return self._concretize_bounds_3d(self.lower, self.upper)
    else:
      raise NotImplementedError('Shape unsupported: {}'.format(shape.as_list()))

  @staticmethod
  def convert(bounds):
    if isinstance(bounds, SymbolicBounds):
      return bounds

    if isinstance(bounds, tf.Tensor):
      bounds = IntervalBounds(bounds, bounds)
    if not isinstance(bounds, IntervalBounds):
      raise ValueError('Cannot convert "{}" to "IntervalBounds"'.format(bounds))

    batch_size = tf.shape(bounds.lower)[0]
    lb = snt.BatchFlatten()(bounds.lower)
    ub = snt.BatchFlatten()(bounds.upper)
    input_size = tf.shape(lb)[1]
    input_shape = bounds.lower.shape[1:]
    output_shape = tf.concat([[input_size], input_shape], axis=0)
    identity = tf.reshape(tf.eye(input_size), output_shape)
    identity = tf.expand_dims(identity, 0)
    identity = tf.tile(identity, [batch_size] + [1] * (len(input_shape) + 1))
    expr = LinearExpression(w=identity, b=tf.zeros_like(bounds.lower),
                            lower=lb, upper=ub)
    return SymbolicBounds(expr, expr)

  def _linear(self, w, b):
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = self._add_expression(
        self._scale_expression(self.lower, w_pos),
        self._scale_expression(self.upper, w_neg)
    )
    lb = self._add_bias(lb, b)
    ub = self._add_expression(
        self._scale_expression(self.lower, w_neg),
        self._scale_expression(self.upper, w_pos)
    )
    ub = self._add_bias(ub, b)
    return SymbolicBounds(lb, ub)

  def _conv2d(self, w, b, padding, strides):
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = self._add_expression(
        self._conv_expression(self.lower, w_pos, padding, strides),
        self._conv_expression(self.upper, w_neg, padding, strides))
    lb = self._add_bias(lb, b)

    ub = self._add_expression(
        self._conv_expression(self.upper, w_pos, padding, strides),
        self._conv_expression(self.lower, w_neg, padding, strides))
    ub = self._add_bias(ub, b)
    return SymbolicBounds(lb, ub)

  def _increasing_monotonic_fn(self, fn, *args):
    if fn.__name__ != 'relu':
      # Fallback to regular interval bound propagation for unsupported
      # operations.
      logging.warn('"%s" is not supported by SymbolicBounds. '
                   'Fallback on IntervalBounds.', fn.__name__)
      interval_bounds = IntervalBounds.convert(self)
      converted_args = [IntervalBounds.convert(b) for b in args]
      interval_bounds = interval_bounds._increasing_monotonic_fn(  # pylint: disable=protected-access
          fn, *converted_args)
      return SymbolicBounds.convert(interval_bounds)

    lb, ub = self.concretize()
    is_ambiguous = tf.logical_and(ub > 0, lb < 0)
    # Ensure denominator is always positive, even when not needed.
    ambiguous_denom = tf.where(is_ambiguous, ub - lb, tf.ones_like(ub))
    scaling_matrix = tf.where(
        is_ambiguous, ub / ambiguous_denom,
        tf.where(lb >= 0, tf.ones_like(lb), tf.zeros_like(lb)))
    bias = tf.where(tf.logical_and(ub > 0, lb < 0), -lb, tf.zeros_like(lb))
    lb_out = LinearExpression(
        w=tf.multiply(tf.expand_dims(scaling_matrix, 1), self.lower.w),
        b=scaling_matrix * self.lower.b, lower=self.lower.lower,
        upper=self.lower.upper)
    ub_out = LinearExpression(
        w=tf.multiply(tf.expand_dims(scaling_matrix, 1), self.upper.w),
        b=scaling_matrix * (self.upper.b + bias), lower=self.upper.lower,
        upper=self.upper.upper)
    return SymbolicBounds(lb_out, ub_out)

  def _batch_flatten(self):
    return SymbolicBounds(self._batch_flatten_expression(self.lower),
                          self._batch_flatten_expression(self.upper))

  # Helper methods.
  @staticmethod
  def _add_bias(expr, b):
    """Add bias b to a linear expression."""
    b = expr.b + b
    return LinearExpression(w=expr.w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _scale_expression(expr, w):
    """Scale a linear expression by w."""
    b = tf.matmul(expr.b, w)
    w = tf.tensordot(expr.w, w, axes=1)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _add_expression(expr_a, expr_b):
    """Add two expression together."""
    return LinearExpression(w=expr_a.w + expr_b.w, b=expr_a.b + expr_b.b,
                            lower=expr_a.lower, upper=expr_b.upper)

  @staticmethod
  def _conv_expression(expr, w, padding, strides):
    """Scale a linear expression by w (through a convolutional layer)."""
    b = tf.nn.convolution(expr.b, w, padding=padding, strides=strides)
    shape = tf.concat([[tf.reduce_prod(tf.shape(expr.w)[:2])],
                       tf.shape(expr.w)[2:]], axis=0)
    w = tf.nn.convolution(tf.reshape(expr.w, shape), w, padding=padding,
                          strides=strides)
    shape = tf.concat([tf.shape(expr.w)[:2], tf.shape(w)[1:]], axis=0)
    w = tf.reshape(w, shape)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _concretize_bounds_1d(lower, upper):
    """Returns lower and upper interval bounds."""
    lb = (tf.einsum('ijk,ij->ik', tf.maximum(lower.w, 0), lower.lower) +
          tf.einsum('ijk,ij->ik', tf.minimum(lower.w, 0), lower.upper) +
          lower.b)
    ub = (tf.einsum('ijk,ij->ik', tf.maximum(upper.w, 0), upper.upper) +
          tf.einsum('ijk,ij->ik', tf.minimum(upper.w, 0), upper.lower) +
          upper.b)
    return lb, ub

  @staticmethod
  def _concretize_bounds_3d(lower, upper):
    """Returns lower and upper interval bounds."""
    lb = (tf.einsum('ijhwc,ij->ihwc', tf.maximum(lower.w, 0), lower.lower) +
          tf.einsum('ijhwc,ij->ihwc', tf.minimum(lower.w, 0), lower.upper) +
          lower.b)
    ub = (tf.einsum('ijhwc,ij->ihwc', tf.maximum(upper.w, 0), upper.upper) +
          tf.einsum('ijhwc,ij->ihwc', tf.minimum(upper.w, 0), upper.lower) +
          upper.b)
    return lb, ub

  @staticmethod
  def _batch_flatten_expression(expr):
    w = snt.FlattenTrailingDimensions(2)(expr.w)
    b = snt.BatchFlatten()(expr.b)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)
