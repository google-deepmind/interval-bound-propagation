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

"""Fast-Lin symbolic bound calculation for common neural network layers.

The Fast-Lin algorithm expresses lower and upper bounds of each layer of
a neural network as a symbolic linear expression in the input neurons,
relaxing the ReLU layers to retain linearity at the expense of tightness.

Reference: "Towards Fast Computation of Certified Robustness for ReLU Networks",
https://arxiv.org/pdf/1804.09699.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from interval_bound_propagation.src import bounds as basic_bounds
from interval_bound_propagation.src import relative_bounds
import sonnet as snt
import tensorflow.compat.v1 as tf


# Holds the linear expressions serving as bounds.
# w: [batch_size, input_size, output_shape] storing the weights.
# b: [batch_size, output_shape] storing the bias.
# lower: [batch_size, input_size] storing the lower bounds on inputs.
# upper: [batch_size, input_size] storing the upper bounds on inputs.
# `lower` and `upper` tensors are always flattened representations of the
# original inputs.
LinearExpression = collections.namedtuple(
    'LinearExpression', ['w', 'b', 'lower', 'upper'])


class SymbolicBounds(basic_bounds.AbstractBounds):
  """Fast-Lin bounds (https://arxiv.org/abs/1804.09699)."""

  def __init__(self, lower, upper):
    super(SymbolicBounds, self).__init__()
    self._lower = lower
    self._upper = upper
    self._prior_bounds = None
    self._concretized = None

  @property
  def lower(self):
    return self._lower

  @property
  def upper(self):
    return self._upper

  @property
  def shape(self):
    return self.lower.b.shape.as_list()

  def concretize(self):
    """Returns lower and upper interval bounds."""
    if self._concretized is None:
      # Construct once and cache.
      lb, ub = self._concretize_bounds(self.lower, self.upper)

      # Apply intersections with prior runs.
      if self._prior_bounds is not None:
        lb = tf.maximum(lb, self._prior_bounds.lower)
        ub = tf.minimum(ub, self._prior_bounds.upper)
      self._concretized = basic_bounds.IntervalBounds(lb, ub)

    return self._concretized

  def with_priors(self, existing_bounds):
    if existing_bounds is not None:
      self._prior_bounds = existing_bounds.concretize()
      # These priors are applied the next time concretize() is called.
      self._concretized = None
    return self

  @classmethod
  def convert(cls, bounds):
    if isinstance(bounds, cls):
      return bounds

    if isinstance(bounds, tf.Tensor):
      bounds = basic_bounds.IntervalBounds(bounds, bounds)
    bounds = bounds.concretize()
    if not isinstance(bounds, basic_bounds.IntervalBounds):
      raise ValueError('Cannot convert "{}" to "SymbolicBounds"'.format(bounds))

    lower, upper = cls._initial_symbolic_bounds(bounds.lower, bounds.upper)
    return cls(lower, upper)

  def apply_linear(self, wrapper, w, b):
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
    return SymbolicBounds(lb, ub).with_priors(wrapper.output_bounds)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = self._add_expression(
        self._conv1d_expression(self.lower, w_pos, padding, stride),
        self._conv1d_expression(self.upper, w_neg, padding, stride))
    lb = self._add_bias(lb, b)

    ub = self._add_expression(
        self._conv1d_expression(self.upper, w_pos, padding, stride),
        self._conv1d_expression(self.lower, w_neg, padding, stride))
    ub = self._add_bias(ub, b)
    return SymbolicBounds(lb, ub).with_priors(wrapper.output_bounds)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = self._add_expression(
        self._conv2d_expression(self.lower, w_pos, padding, strides),
        self._conv2d_expression(self.upper, w_neg, padding, strides))
    lb = self._add_bias(lb, b)

    ub = self._add_expression(
        self._conv2d_expression(self.upper, w_pos, padding, strides),
        self._conv2d_expression(self.lower, w_neg, padding, strides))
    ub = self._add_bias(ub, b)
    return SymbolicBounds(lb, ub).with_priors(wrapper.output_bounds)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    if fn.__name__ != 'relu':
      # Fallback to regular interval bound propagation for unsupported
      # operations.
      logging.warn('"%s" is not supported by SymbolicBounds. '
                   'Fallback on IntervalBounds.', fn.__name__)
      interval_bounds = basic_bounds.IntervalBounds.convert(self)
      converted_args = [basic_bounds.IntervalBounds.convert(b) for b in args]
      interval_bounds = interval_bounds._increasing_monotonic_fn(  # pylint: disable=protected-access
          fn, *converted_args)
      return self.convert(interval_bounds)

    concrete = self.concretize()
    lb, ub = concrete.lower, concrete.upper
    is_ambiguous = tf.logical_and(ub > 0, lb < 0)
    # Ensure denominator is always positive, even when not needed.
    ambiguous_denom = tf.where(is_ambiguous, ub - lb, tf.ones_like(ub))
    scale = tf.where(
        is_ambiguous, ub / ambiguous_denom,
        tf.where(lb >= 0, tf.ones_like(lb), tf.zeros_like(lb)))
    bias = tf.where(is_ambiguous, -lb, tf.zeros_like(lb))
    lb_out = LinearExpression(
        w=tf.expand_dims(scale, 1) * self.lower.w,
        b=scale * self.lower.b,
        lower=self.lower.lower, upper=self.lower.upper)
    ub_out = LinearExpression(
        w=tf.expand_dims(scale, 1) * self.upper.w,
        b=scale * (self.upper.b + bias),
        lower=self.upper.lower, upper=self.upper.upper)
    return SymbolicBounds(lb_out, ub_out).with_priors(wrapper.output_bounds)

  def apply_batch_reshape(self, wrapper, shape):
    return SymbolicBounds(self._batch_reshape_expression(self.lower, shape),
                          self._batch_reshape_expression(self.upper, shape)
                         ).with_priors(wrapper.output_bounds)

  # Helper methods.
  @staticmethod
  def _add_bias(expr, b):
    """Add bias b to a linear expression."""
    if b is None:
      return expr
    return LinearExpression(w=expr.w, b=expr.b + b,
                            lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _add_expression(expr_a, expr_b):
    """Add two expression together."""
    return LinearExpression(w=expr_a.w + expr_b.w, b=expr_a.b + expr_b.b,
                            lower=expr_a.lower, upper=expr_b.upper)

  @staticmethod
  def _scale_expression(expr, w):
    """Scale a linear expression by w."""
    b = tf.matmul(expr.b, w)
    w = tf.tensordot(expr.w, w, axes=1)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _conv1d_expression(expr, w, padding, stride):
    """Scale a linear expression by w (through a convolutional layer)."""
    b = tf.nn.conv1d(expr.b, w, padding=padding, stride=stride)
    shape = tf.concat([[tf.reduce_prod(tf.shape(expr.w)[:2])],
                       tf.shape(expr.w)[2:]], axis=0)
    w = tf.nn.conv1d(tf.reshape(expr.w, shape), w, padding=padding,
                     stride=stride)
    shape = tf.concat([tf.shape(expr.w)[:2], tf.shape(w)[1:]], axis=0)
    w = tf.reshape(w, shape)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _conv2d_expression(expr, w, padding, strides):
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
  def _batch_reshape_expression(expr, shape):
    w = snt.BatchReshape(shape, preserve_dims=2)(expr.w)
    b = snt.BatchReshape(shape)(expr.b)
    return LinearExpression(w=w, b=b, lower=expr.lower, upper=expr.upper)

  @staticmethod
  def _concretize_bounds(lower, upper):
    """Returns lower and upper interval bounds."""
    if len(lower.b.shape) == 2:
      equation = 'ijk,ij->ik'
    elif len(lower.b.shape) == 3:
      equation = 'ijnc,ij->inc'
    elif len(lower.b.shape) == 4:
      equation = 'ijhwc,ij->ihwc'
    else:
      raise NotImplementedError('Shape unsupported: {}'.format(lower.b.shape))

    lb = (tf.einsum(equation, tf.maximum(lower.w, 0), lower.lower) +
          tf.einsum(equation, tf.minimum(lower.w, 0), lower.upper) +
          lower.b)
    ub = (tf.einsum(equation, tf.maximum(upper.w, 0), upper.upper) +
          tf.einsum(equation, tf.minimum(upper.w, 0), upper.lower) +
          upper.b)
    return lb, ub

  @staticmethod
  def _initial_symbolic_bounds(lb, ub):
    """Returns symbolic bounds for the given interval bounds."""
    batch_size = tf.shape(lb)[0]
    input_shape = lb.shape[1:]
    zero = tf.zeros_like(lb)
    lb = snt.BatchFlatten()(lb)
    ub = snt.BatchFlatten()(ub)
    input_size = tf.shape(lb)[1]
    output_shape = tf.concat([[input_size], input_shape], axis=0)
    identity = tf.reshape(tf.eye(input_size), output_shape)
    identity = tf.expand_dims(identity, 0)
    identity = tf.tile(identity, [batch_size] + [1] * (len(input_shape) + 1))
    expr = LinearExpression(w=identity, b=zero,
                            lower=lb, upper=ub)
    return expr, expr


class RelativeSymbolicBounds(SymbolicBounds):
  """Relative-to-nominal variant of Fast-Lin bounds."""

  def __init__(self, lower_offset, upper_offset, nominal):
    super(RelativeSymbolicBounds, self).__init__(lower_offset, upper_offset)
    self._nominal = nominal

  def concretize(self):
    """Returns lower and upper interval bounds."""
    if self._concretized is None:
      # Construct once and cache.
      lb_offset, ub_offset = self._concretize_bounds(self.lower, self.upper)

      # Apply intersections with prior runs.
      if self._prior_bounds is not None:
        lb_offset = tf.maximum(lb_offset, self._prior_bounds.lower_offset)
        ub_offset = tf.minimum(ub_offset, self._prior_bounds.upper_offset)
      self._concretized = relative_bounds.RelativeIntervalBounds(
          lb_offset, ub_offset, self._nominal)

    return self._concretized

  @classmethod
  def convert(cls, bounds):
    if isinstance(bounds, cls):
      return bounds

    if isinstance(bounds, tf.Tensor):
      bounds = relative_bounds.RelativeIntervalBounds(
          tf.zeros_like(bounds), tf.zeros_like(bounds), bounds)
    bounds = bounds.concretize()
    if not isinstance(bounds, relative_bounds.RelativeIntervalBounds):
      raise ValueError(
          'Cannot convert "{}" to "RelativeSymbolicBounds"'.format(bounds))

    lower, upper = cls._initial_symbolic_bounds(bounds.lower_offset,
                                                bounds.upper_offset)
    return cls(lower, upper, bounds.nominal)

  def apply_linear(self, wrapper, w, b):
    bounds_out = super(RelativeSymbolicBounds, self).apply_linear(
        wrapper, w, b=None)

    nominal_out = tf.matmul(self._nominal, w)
    if b is not None:
      nominal_out += b

    return RelativeSymbolicBounds(
        bounds_out.lower, bounds_out.upper, nominal_out).with_priors(
            wrapper.output_bounds)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    bounds_out = super(RelativeSymbolicBounds, self).apply_conv1d(
        wrapper, w, b=None, padding=padding, stride=stride)

    nominal_out = tf.nn.conv1d(self._nominal, w,
                               padding=padding, stride=stride)
    if b is not None:
      nominal_out += b

    return RelativeSymbolicBounds(
        bounds_out.lower, bounds_out.upper, nominal_out).with_priors(
            wrapper.output_bounds)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    bounds_out = super(RelativeSymbolicBounds, self).apply_conv2d(
        wrapper, w, b=None, padding=padding, strides=strides)

    nominal_out = tf.nn.convolution(self._nominal, w,
                                    padding=padding, strides=strides)
    if b is not None:
      nominal_out += b

    return RelativeSymbolicBounds(
        bounds_out.lower, bounds_out.upper, nominal_out).with_priors(
            wrapper.output_bounds)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    if fn.__name__ != 'relu':
      # Fallback to regular interval bound propagation for unsupported
      # operations.
      logging.warn('"%s" is not supported by RelativeSymbolicBounds. '
                   'Fallback on RelativeIntervalBounds.', fn.__name__)
      interval_bounds = relative_bounds.RelativeIntervalBounds.convert(self)
      converted_args = [relative_bounds.RelativeIntervalBounds.convert(b)
                        for b in args]
      interval_bounds = interval_bounds._increasing_monotonic_fn(  # pylint: disable=protected-access
          fn, *converted_args)
      return self.convert(interval_bounds)

    concrete = self.concretize()
    lb, ub = concrete.lower_offset, concrete.upper_offset
    is_ambiguous = tf.logical_and(ub > -self._nominal, lb < -self._nominal)
    # Ensure denominator is always positive, even when not needed.
    ambiguous_denom = tf.where(is_ambiguous, ub - lb, tf.ones_like(ub))
    scale = tf.where(
        is_ambiguous, (self._nominal + ub) / ambiguous_denom,
        tf.where(lb >= -self._nominal, tf.ones_like(lb), tf.zeros_like(lb)))
    scale_complement = tf.where(
        is_ambiguous, -(self._nominal + lb) / ambiguous_denom,
        tf.where(lb >= -self._nominal, tf.zeros_like(lb), tf.ones_like(lb)))
    # Need lb_out.b = scale * (nom_in + lb_in.b) - nom_out
    # and ub_out.b = scale * (nom_in + ub_in.b - min(nom_in + lb, 0)) - nom_out
    lower_bias = (scale * (tf.minimum(self._nominal, 0.)) +
                  scale_complement * tf.minimum(-self._nominal, 0.))
    upper_bias = (scale * tf.maximum(tf.minimum(-self._nominal, 0.) - lb,
                                     tf.minimum(self._nominal, 0.)) +
                  scale_complement * tf.minimum(-self._nominal, 0.))
    lb_out = LinearExpression(
        w=tf.expand_dims(scale, 1) * self.lower.w,
        b=scale * self.lower.b + lower_bias,
        lower=self.lower.lower, upper=self.lower.upper)
    ub_out = LinearExpression(
        w=tf.expand_dims(scale, 1) * self.upper.w,
        b=scale * self.upper.b + upper_bias,
        lower=self.upper.lower, upper=self.upper.upper)

    nominal_out = tf.nn.relu(self._nominal)
    return RelativeSymbolicBounds(
        lb_out, ub_out, nominal_out).with_priors(wrapper.output_bounds)

  def apply_batch_reshape(self, wrapper, shape):
    bounds_out = super(RelativeSymbolicBounds, self).apply_batch_reshape(
        wrapper, shape)
    nominal_out = snt.BatchReshape(shape)(self._nominal)
    return RelativeSymbolicBounds(
        bounds_out.lower, bounds_out.upper, nominal_out).with_priors(
            wrapper.output_bounds)
